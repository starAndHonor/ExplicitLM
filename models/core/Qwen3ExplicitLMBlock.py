"""
Qwen3ExplicitLMBlock: 
"""

from typing import Dict, Tuple, Union, Optional, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Cache,
)
from transformers.utils import TransformersKwargs
from typing import Unpack

from models.memory_bank.MemoryGate import MemoryGate
from models.memory_bank.GatedMemoryFusion import GatedMemoryFusion
from models.layers.RMSNorm import RMSNorm
from models.core.Qwen3ExplicitDecoderLayer import Qwen3ExplicitDecoderLayer

logger = logging.getLogger(__name__)


class MemorySelectionResult(NamedTuple):
    selected_memory: torch.Tensor
    selection_weights: torch.Tensor
    selected_indices: torch.Tensor
    actual_memory_indices: torch.Tensor
    similarity_scores: torch.Tensor
    selected_similarities: torch.Tensor


class Qwen3ExplicitLMBlock(nn.Module):

    def __init__(
        self, 
        config: Qwen3Config, 
        layer_idx: int, 
        memory_cfg: dict, 
        shared_memory_gate: Optional[MemoryGate] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.memory_cfg = memory_cfg
        self.hidden_size = config.hidden_size
        self.use_moe = memory_cfg.get("use_moe", False)
        
        # 基础Transformer层（融合位置前移：在 Attention 与 FFN 之间插入）
        self.qwen3_decoder = Qwen3ExplicitDecoderLayer(config, layer_idx)
        
        if not self.use_moe:
            self._init_memory_components(config, memory_cfg, shared_memory_gate)
        
        # 缓存相关
        self._cached_memory_embeddings = None
        self._cached_normalized_memories = None
        
    def _init_memory_components(
        self, 
        config: Qwen3Config, 
        memory_cfg: dict, 
        shared_memory_gate: MemoryGate
    ):
        """初始化记忆相关组件"""
        if shared_memory_gate is None:
            raise ValueError("shared_memory_gate must be provided when use_moe=False")
        
        self.memory_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.memory_gate = shared_memory_gate
        
        # 融合模块
        memory_cfg_with_dim = memory_cfg.copy()
        memory_cfg_with_dim["dim"] = config.hidden_size
        self.gated_memory_fusion = GatedMemoryFusion(memory_cfg_with_dim)
        
        # Gumbel温度
        self.gumbel_temperature = memory_cfg.get("gumbel_temperature", 1.0)
        
        # 残差缩放（简化版本）
        scale_init = memory_cfg.get("memory_residual_scale_init", 0.05)
        trainable = memory_cfg.get("memory_residual_scale_trainable", True)
        self.log_residual_scale = nn.Parameter(
            torch.tensor(scale_init).clamp(1e-6, 1-1e-6).logit()
        )
        self.log_residual_scale.requires_grad = trainable
        
        # Pad token ID
        self.pad_token_id = getattr(config, 'pad_token_id', 0) or 0
        
        # 嵌入批次大小（用于大规模候选）
        self.embed_batch_size = memory_cfg.get("embed_batch_size", 64)

    @torch.no_grad()
    def precompute_memory_cache(
        self, 
        memory_bank: torch.Tensor, 
        tok_embeddings: nn.Embedding,
        device: torch.device
    ):
        """预计算并缓存记忆库的嵌入表示（推理优化）"""
        if self.use_moe:
            return
        
        logger.info(f"Layer {self.layer_idx}: Precomputing memory cache...")
        
        # 将memory_bank移到目标设备
        memory_bank = memory_bank.to(device)
        
        # 批量计算所有记忆的嵌入
        num_memories = memory_bank.size(0)
        all_embeddings = []
        
        for i in range(0, num_memories, self.embed_batch_size):
            end_idx = min(i + self.embed_batch_size, num_memories)
            batch_ids = memory_bank[i:end_idx]
            batch_embeds = tok_embeddings(batch_ids).mean(dim=1)
            all_embeddings.append(batch_embeds)
        
        self._cached_memory_embeddings = torch.cat(all_embeddings, dim=0)
        
        # 预计算归一化版本
        self._cached_normalized_memories = F.normalize(
            self._cached_memory_embeddings, p=2, dim=-1
        )
        
        logger.info(f"Layer {self.layer_idx}: Memory cache ready")

    def clear_cache(self):
        """清除缓存（训练时或显存不足时使用）"""
        self._cached_memory_embeddings = None
        self._cached_normalized_memories = None

    def gumbel_softmax_selection(
        self,
        similarity_scores: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """优化的Gumbel-Softmax选择"""
        # 推理时直接使用argmax，无需Gumbel噪声
        if not self.training:
            indices = similarity_scores.argmax(dim=-1)
            one_hot = F.one_hot(indices, similarity_scores.size(-1)).float()
            return one_hot, indices
        
        # 训练时使用Gumbel-Softmax
        # 使用更高效的Gumbel噪声生成
        gumbel = -torch.empty_like(similarity_scores).exponential_().log()
        logits = (similarity_scores + gumbel) / temperature
        soft_weights = F.softmax(logits, dim=-1)
        
        if hard:
            indices = soft_weights.argmax(dim=-1)
            hard_weights = F.one_hot(indices, soft_weights.size(-1)).float()
            # Straight-through estimator
            selection_weights = hard_weights.detach() + soft_weights - soft_weights.detach()
            return selection_weights, indices
        
        return soft_weights, soft_weights.argmax(dim=-1)

    def _get_candidate_embeddings(
        self,
        candidate_indices: torch.Tensor,
        memory_bank: torch.Tensor,
        tok_embeddings: nn.Embedding,
    ) -> torch.Tensor:
        """获取候选记忆的嵌入（带缓存优化）"""
        bsz, seq_len, num_candidates = candidate_indices.shape
        
        # 如果有缓存，直接使用
        if self._cached_memory_embeddings is not None:
            flat_indices = candidate_indices.reshape(-1)
            candidate_embeds = self._cached_memory_embeddings[flat_indices]
            return candidate_embeds.reshape(bsz, seq_len, num_candidates, self.hidden_size)
        
        # 无缓存：动态计算（训练模式）
        flat_indices = candidate_indices.reshape(-1)
        
        # 优化：只获取需要的token ids
        if memory_bank.device != candidate_indices.device:
            # 只传输需要的部分
            candidate_token_ids = memory_bank[flat_indices.cpu()].to(
                candidate_indices.device, non_blocking=True
            )
        else:
            candidate_token_ids = memory_bank[flat_indices]
        
        # 批量计算嵌入
        num_total = candidate_token_ids.size(0)
        if num_total <= self.embed_batch_size:
            # 小batch直接计算
            embeddings = tok_embeddings(candidate_token_ids).mean(dim=1)
        else:
            # 大batch分批计算（使用checkpoint节省显存）
            embeddings = self._batch_embed_tokens(candidate_token_ids, tok_embeddings)
        
        return embeddings.reshape(bsz, seq_len, num_candidates, self.hidden_size)

    def _batch_embed_tokens(
        self, 
        token_ids: torch.Tensor, 
        tok_embeddings: nn.Embedding
    ) -> torch.Tensor:
        """批量计算token嵌入（大batch优化）"""
        num_total = token_ids.size(0)
        embeddings_list = []
        
        for i in range(0, num_total, self.embed_batch_size):
            end_idx = min(i + self.embed_batch_size, num_total)
            batch_ids = token_ids[i:end_idx]
            batch_embeds = tok_embeddings(batch_ids).mean(dim=1)
            embeddings_list.append(batch_embeds)
        
        return torch.cat(embeddings_list, dim=0)

    def _compute_similarity_scores(
        self,
        h_for_memory: torch.Tensor,
        candidate_memories: torch.Tensor,
        candidate_indices: torch.Tensor,
        memory_bank: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算相似度分数（带有效性过滤）"""
        bsz, seq_len, num_candidates, _ = candidate_memories.shape
        
        # 计算余弦相似度
        if self._cached_normalized_memories is not None:
            # 使用预计算的归一化记忆
            h_normalized = F.normalize(h_for_memory, p=2, dim=-1)
            flat_indices = candidate_indices.reshape(-1)
            candidate_normalized = self._cached_normalized_memories[flat_indices]
            candidate_normalized = candidate_normalized.reshape(bsz, seq_len, num_candidates, -1)
            similarity_scores = torch.einsum('bsd,bsnd->bsn', h_normalized, candidate_normalized)
        else:
            # 动态计算归一化
            h_normalized = F.normalize(h_for_memory, p=2, dim=-1)
            candidate_normalized = F.normalize(candidate_memories, p=2, dim=-1)
            similarity_scores = torch.einsum('bsd,bsnd->bsn', h_normalized, candidate_normalized)
        
        # 构建有效性mask
        candidate_valid = self._get_validity_mask(
            candidate_indices, memory_bank, valid_mask
        )
        
        # 过滤无效候选
        similarity_scores = similarity_scores.masked_fill(~candidate_valid, -1e9)
        
        return similarity_scores

    def _get_validity_mask(
        self,
        candidate_indices: torch.Tensor,
        memory_bank: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取候选项的有效性mask"""
        bsz, seq_len, num_candidates = candidate_indices.shape
        device = candidate_indices.device
        
        # 检查内容是否全为pad
        flat_indices = candidate_indices.reshape(-1)
        if memory_bank.device != device:
            candidate_tokens = memory_bank[flat_indices.cpu()].to(device, non_blocking=True)
        else:
            candidate_tokens = memory_bank[flat_indices]
        
        is_not_all_pad = ~(candidate_tokens == self.pad_token_id).all(dim=-1)
        is_not_all_pad = is_not_all_pad.reshape(bsz, seq_len, num_candidates)
        
        # 如果有valid_mask，同时检查
        if valid_mask is not None:
            if valid_mask.device != device:
                valid_mask = valid_mask.to(device, non_blocking=True)
            candidate_valid_from_mask = valid_mask[candidate_indices]
            return candidate_valid_from_mask & is_not_all_pad
        
        return is_not_all_pad

    def _select_memory(
        self,
        h_for_memory: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_scores: torch.Tensor,
        memory_bank: torch.Tensor,
        tok_embeddings: nn.Embedding,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> MemorySelectionResult:
        """记忆选择流程（核心逻辑）"""
        # 1. 获取候选嵌入
        candidate_memories = self._get_candidate_embeddings(
            candidate_indices, memory_bank, tok_embeddings
        )
        
        # 2. 计算相似度
        similarity_scores = self._compute_similarity_scores(
            h_for_memory, candidate_memories, candidate_indices, 
            memory_bank, valid_mask
        )
        
        # 3. Gumbel选择
        selection_weights, selected_indices = self.gumbel_softmax_selection(
            similarity_scores, self.gumbel_temperature, hard=True
        )
        
        # 4. 获取实际选中的记忆库索引
        actual_memory_indices = candidate_indices.gather(
            dim=-1, index=selected_indices.unsqueeze(-1)
        ).squeeze(-1)
        
        # 5. 获取选中记忆的相似度
        selected_similarities = similarity_scores.gather(
            dim=-1, index=selected_indices.unsqueeze(-1)
        ).squeeze(-1)
        
        # 6. 加权求和得到选中的记忆
        selection_weights = selection_weights.to(candidate_memories.dtype)
        selected_memory = torch.einsum(
            'bsn,bsnd->bsd', selection_weights, candidate_memories
        )
        
        return MemorySelectionResult(
            selected_memory=selected_memory,
            selection_weights=selection_weights,
            selected_indices=selected_indices,
            actual_memory_indices=actual_memory_indices,
            similarity_scores=similarity_scores,
            selected_similarities=selected_similarities,
        )

    def _fuse_memory(
        self,
        hidden_states: torch.Tensor,
        h_for_memory: torch.Tensor,
        selection_result: MemorySelectionResult,
    ) -> torch.Tensor:
        """记忆融合"""
        # 使用GatedMemoryFusion
        memory_output = self.gated_memory_fusion(
            h_for_memory,
            selection_result.selected_memory,
            similarity_scores=selection_result.selected_similarities,
        )
        
        # 应用全局残差缩放
        residual_scale = torch.sigmoid(self.log_residual_scale).to(memory_output.dtype)
        memory_output = residual_scale * memory_output
        
        # 残差连接
        return hidden_states + memory_output

    def _compute_memory_loss(
        self,
        h_for_memory: torch.Tensor,
        selection_result: MemorySelectionResult,
        memory_bank: torch.Tensor,
        tok_embeddings: nn.Embedding,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算记忆检索损失"""
        return self.memory_gate.compute_loss_single_target(
            query_embedding=h_for_memory,
            selected_knowledge_embedding=selection_result.selected_memory,
            memory_bank=memory_bank,
            selected_index=selection_result.actual_memory_indices,
            tok_embeddings=tok_embeddings,
            valid_mask=valid_mask,
            pad_token_id=self.pad_token_id,
        )

    def _compute_stats(
        self,
        selection_result: MemorySelectionResult,
    ) -> Tuple[Dict[str, float], Dict[str, Union[torch.Tensor, float]]]:
        """计算统计信息（仅训练时）"""
        if not self.training:
            return {}, {}
        
        # Layer统计
        layer_stats = self._compute_selection_stats(
            selection_result.selection_weights,
            selection_result.actual_memory_indices,
        )
        
        # Cosine统计
        cosine_stats = {
            "similarity_scores": selection_result.similarity_scores,
            "selected_similarities": selection_result.selected_similarities,
            "actual_selected_indices": selection_result.actual_memory_indices,
            "avg_similarity": selection_result.similarity_scores.mean().item(),
            "max_similarity": selection_result.similarity_scores.max().item(),
            "min_similarity": selection_result.similarity_scores.min().item(),
            "selected_avg_similarity": selection_result.selected_similarities.mean().item(),
            "selection_entropy": -torch.sum(
                selection_result.selection_weights * 
                torch.log(selection_result.selection_weights + 1e-10), 
                dim=-1
            ).mean().item(),
        }
        
        return layer_stats, cosine_stats

    def _compute_selection_stats(
        self,
        selection_weights: torch.Tensor,
        actual_indices: torch.Tensor,
    ) -> Dict[str, float]:
        """计算选择统计"""
        device = actual_indices.device
        knowledge_num = self.memory_cfg["knowledge_num"]
        
        # 统计每个记忆被选中的次数
        memory_counts = torch.zeros(knowledge_num, device=device, dtype=selection_weights.dtype)
        flat_weights = selection_weights.reshape(-1)
        flat_indices = actual_indices.reshape(-1)
        memory_counts.scatter_add_(0, flat_indices, flat_weights)
        
        with torch.no_grad():
            counts_fp32 = memory_counts.float()
            coverage_rate = (counts_fp32 > 0.01).float().mean().item()
            top10_threshold = torch.quantile(counts_fp32, 0.9)
            hot_memories = (counts_fp32 >= top10_threshold).sum().item()
            dead_memories = (counts_fp32 < 0.01).sum().item()
            
            return {
                "coverage_rate": coverage_rate,
                "hot_memories": hot_memories,
                "dead_memories": dead_memories,
                "selection_variance": counts_fp32.var().item(),
                "max_selections": counts_fp32.max().item(),
                "min_selections": counts_fp32.min().item(),
            }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        memory_bank: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        tok_embeddings: Optional[nn.Embedding] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]]]:
        """
        主前向传播
        
        Returns:
            (output, similarity_loss, layer_stats, cosine_stats)
        """
        if self.use_moe:
            # MoE模式：不启用记忆融合
            hidden_states = self.qwen3_decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            zero_loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=False)
            return hidden_states, zero_loss, {}, {}

        selection_cache: Dict[str, Union[torch.Tensor, MemorySelectionResult]] = {}

        def memory_fusion(attn_hidden_states: torch.Tensor) -> torch.Tensor:
            # 记忆检索在 Attention 与 FFN 之间进行
            h_for_memory = self.memory_norm(attn_hidden_states)
            candidate_indices, candidate_scores = self.memory_gate(h_for_memory)
            selection_result = self._select_memory(
                h_for_memory, candidate_indices, candidate_scores,
                memory_bank, tok_embeddings, valid_mask
            )
            selection_cache["h_for_memory"] = h_for_memory
            selection_cache["selection_result"] = selection_result
            return self._fuse_memory(attn_hidden_states, h_for_memory, selection_result)

        # 1. Transformer前向（Attention -> 记忆融合 -> FFN，融合位置前移）
        hidden_states = self.qwen3_decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            memory_fusion=memory_fusion,
            **kwargs,
        )

        # 2. 计算损失与统计（基于 Attention 后的状态）
        selection_result = selection_cache["selection_result"]
        h_for_memory = selection_cache["h_for_memory"]
        similarity_loss = self._compute_memory_loss(
            h_for_memory, selection_result, memory_bank,
            tok_embeddings, valid_mask
        )
        layer_stats, cosine_stats = self._compute_stats(selection_result)

        return hidden_states, similarity_loss, layer_stats, cosine_stats
