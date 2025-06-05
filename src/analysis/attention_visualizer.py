"""
Attention Visualization Tools for MMTD Models
MMTD 모델의 Attention 시각화 도구
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 10)

class AttentionVisualizer:
    """Attention 분석 결과 시각화 클래스"""
    
    def __init__(self, figsize=(15, 10), dpi=100):
        """
        Args:
            figsize: 기본 그림 크기
            dpi: 해상도
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # 컬러맵 설정
        self.text_cmap = plt.cm.Reds
        self.image_cmap = plt.cm.Blues
        self.cross_modal_cmap = plt.cm.Purples
        
    def visualize_text_attention(self, tokens: List[str], 
                               token_importance: List[Dict[str, Any]],
                               title: str = "텍스트 토큰 Attention 중요도",
                               save_path: Optional[str] = None) -> plt.Figure:
        """텍스트 토큰별 attention 시각화"""
        
        if not token_importance:
            print("❌ 토큰 중요도 데이터가 없습니다.")
            return None
        
        # 데이터 준비
        top_tokens = token_importance[:15]  # 상위 15개만
        token_names = [item['token'] for item in top_tokens]
        importance_scores = [item['combined_importance'] for item in top_tokens]
        
        # 정규화
        max_score = max(importance_scores) if importance_scores else 1
        normalized_scores = [score / max_score for score in importance_scores]
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 바 차트
        colors = self.text_cmap(np.array(normalized_scores))
        bars = ax1.barh(range(len(token_names)), importance_scores, color=colors)
        
        ax1.set_yticks(range(len(token_names)))
        ax1.set_yticklabels(token_names)
        ax1.set_xlabel('Attention 중요도')
        ax1.set_title('토큰별 중요도 순위')
        ax1.grid(True, alpha=0.3)
        
        # 값 레이블 추가
        for i, (bar, score) in enumerate(zip(bars, importance_scores)):
            ax1.text(bar.get_width() + max(importance_scores) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', va='center', fontsize=9)
        
        # 2. 텍스트 하이라이팅 시각화
        self._visualize_text_highlighting(ax2, tokens, token_importance)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 텍스트 attention 시각화 저장: {save_path}")
        
        return fig
    
    def _visualize_text_highlighting(self, ax, tokens: List[str], 
                                   token_importance: List[Dict[str, Any]]):
        """텍스트 하이라이팅 방식으로 중요도 표시"""
        
        # 토큰별 중요도 매핑
        importance_dict = {item['token']: item['combined_importance'] for item in token_importance}
        max_importance = max(importance_dict.values()) if importance_dict else 1
        
        # 텍스트 재구성 및 시각화
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')
        ax.set_title('텍스트 중요도 하이라이팅')
        
        x_pos = 0.1
        y_pos = 2.5
        line_height = 0.8
        
        for i, token in enumerate(tokens[:50]):  # 처음 50개 토큰만
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            # 토큰 정리 (##제거)
            display_token = token.replace('##', '')
            if not display_token.strip():
                continue
            
            # 중요도에 따른 색상
            importance = importance_dict.get(token, 0)
            alpha = min(importance / max_importance, 1.0) if max_importance > 0 else 0
            color = self.text_cmap(alpha)
            
            # 텍스트 표시
            text_width = len(display_token) * 0.15
            
            if x_pos + text_width > 9.5:  # 줄바꿈
                x_pos = 0.1
                y_pos -= line_height
                
            if y_pos < 0.5:  # 화면 벗어남
                break
            
            # 배경 박스
            bbox = dict(boxstyle="round,pad=0.02", facecolor=color, alpha=0.8)
            ax.text(x_pos, y_pos, display_token, fontsize=10, 
                   bbox=bbox, ha='left', va='center')
            
            x_pos += text_width + 0.1
    
    def visualize_image_attention(self, image: torch.Tensor,
                                patch_importance: List[Dict[str, Any]],
                                patch_size: int = 16,
                                title: str = "이미지 패치 Attention 중요도",
                                save_path: Optional[str] = None) -> plt.Figure:
        """이미지 패치별 attention 시각화"""
        
        if not patch_importance:
            print("❌ 패치 중요도 데이터가 없습니다.")
            return None
        
        # 이미지 준비
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0]  # 배치 차원 제거
            image_np = image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        H, W = image_np.shape[:2]
        patch_h, patch_w = H // patch_size, W // patch_size
        
        # Attention 맵 생성
        attention_map = np.zeros((patch_h, patch_w))
        max_importance = max([p['combined_importance'] for p in patch_importance])
        
        for patch_info in patch_importance:
            coord = patch_info['coordinates']
            importance = patch_info['combined_importance']
            
            if coord[0] < patch_h and coord[1] < patch_w:
                attention_map[coord[0], coord[1]] = importance / max_importance
        
        # Attention 맵을 원본 이미지 크기로 리사이징
        attention_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 원본 이미지
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('원본 이미지')
        axes[0, 0].axis('off')
        
        # 2. Attention 히트맵
        im = axes[0, 1].imshow(attention_resized, cmap=self.image_cmap, alpha=0.8)
        axes[0, 1].set_title('Attention 히트맵')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. 오버레이
        axes[1, 0].imshow(image_np)
        axes[1, 0].imshow(attention_resized, cmap=self.image_cmap, alpha=0.6)
        axes[1, 0].set_title('오버레이 (원본 + Attention)')
        axes[1, 0].axis('off')
        
        # 4. 상위 패치 중요도 바 차트
        top_patches = patch_importance[:10]
        patch_labels = [f"({p['coordinates'][0]},{p['coordinates'][1]})" for p in top_patches]
        patch_scores = [p['combined_importance'] for p in top_patches]
        
        bars = axes[1, 1].bar(range(len(patch_labels)), patch_scores, 
                             color=self.image_cmap(np.linspace(0.8, 0.3, len(patch_labels))))
        axes[1, 1].set_title('상위 패치 중요도')
        axes[1, 1].set_xlabel('패치 좌표')
        axes[1, 1].set_ylabel('중요도')
        axes[1, 1].set_xticks(range(len(patch_labels)))
        axes[1, 1].set_xticklabels(patch_labels, rotation=45)
        
        # 값 레이블
        for bar, score in zip(bars, patch_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(patch_scores) * 0.01,
                           f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 이미지 attention 시각화 저장: {save_path}")
        
        return fig
    
    def visualize_cross_modal_attention(self, cross_modal_attention: Dict[str, torch.Tensor],
                                      tokens: List[str],
                                      title: str = "Cross-Modal Attention 분석",
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Cross-modal attention 시각화"""
        
        if not cross_modal_attention:
            print("❌ Cross-modal attention 데이터가 없습니다.")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Text-to-Image Attention
        if 'text_to_image' in cross_modal_attention:
            text_to_image = cross_modal_attention['text_to_image'][0].cpu().numpy()
            im1 = axes[0, 0].imshow(text_to_image, cmap=self.cross_modal_cmap, aspect='auto')
            axes[0, 0].set_title('Text → Image Attention')
            axes[0, 0].set_xlabel('이미지 패치')
            axes[0, 0].set_ylabel('텍스트 토큰')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Image-to-Text Attention
        if 'image_to_text' in cross_modal_attention:
            image_to_text = cross_modal_attention['image_to_text'][0].cpu().numpy()
            im2 = axes[0, 1].imshow(image_to_text, cmap=self.cross_modal_cmap, aspect='auto')
            axes[0, 1].set_title('Image → Text Attention')
            axes[0, 1].set_xlabel('텍스트 토큰')
            axes[0, 1].set_ylabel('이미지 패치')
            plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. CLS Token Attention
        cls_text_attention = cross_modal_attention.get('cls_to_text', torch.zeros(1, 1, 10))
        cls_image_attention = cross_modal_attention.get('cls_to_image', torch.zeros(1, 1, 10))
        
        cls_text = cls_text_attention[0, 0].cpu().numpy()
        cls_image = cls_image_attention[0, 0].cpu().numpy()
        
        axes[0, 2].bar(['Text', 'Image'], [cls_text.sum(), cls_image.sum()], 
                      color=['red', 'blue'], alpha=0.7)
        axes[0, 2].set_title('CLS Token Attention Distribution')
        axes[0, 2].set_ylabel('Attention Sum')
        
        # 4. Attention 강도 분포
        if 'text_to_image' in cross_modal_attention:
            text_to_image_flat = text_to_image.flatten()
            axes[1, 0].hist(text_to_image_flat, bins=50, alpha=0.7, color='purple')
            axes[1, 0].set_title('Text→Image Attention 분포')
            axes[1, 0].set_xlabel('Attention 값')
            axes[1, 0].set_ylabel('빈도')
        
        # 5. 모달리티 균형 분석
        if 'text_to_image' in cross_modal_attention and 'image_to_text' in cross_modal_attention:
            text_strength = cross_modal_attention['text_to_image'].mean().item()
            image_strength = cross_modal_attention['image_to_text'].mean().item()
            
            labels = ['Text Modality', 'Image Modality']
            sizes = [text_strength, image_strength]
            colors = ['lightcoral', 'lightskyblue']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('모달리티 기여도 균형')
        
        # 6. 상호작용 강도 매트릭스
        if len(cross_modal_attention) >= 2:
            interaction_matrix = np.zeros((2, 2))
            
            if 'text_to_image' in cross_modal_attention:
                interaction_matrix[0, 1] = cross_modal_attention['text_to_image'].mean().item()
            if 'image_to_text' in cross_modal_attention:
                interaction_matrix[1, 0] = cross_modal_attention['image_to_text'].mean().item()
            if 'text_to_text' in cross_modal_attention:
                interaction_matrix[0, 0] = cross_modal_attention['text_to_text'].mean().item()
            if 'image_to_image' in cross_modal_attention:
                interaction_matrix[1, 1] = cross_modal_attention['image_to_image'].mean().item()
            
            im3 = axes[1, 2].imshow(interaction_matrix, cmap='RdYlBu_r', annot=True)
            axes[1, 2].set_title('상호작용 강도 매트릭스')
            axes[1, 2].set_xticks([0, 1])
            axes[1, 2].set_yticks([0, 1])
            axes[1, 2].set_xticklabels(['Text', 'Image'])
            axes[1, 2].set_yticklabels(['Text', 'Image'])
            
            # 값 표시
            for i in range(2):
                for j in range(2):
                    axes[1, 2].text(j, i, f'{interaction_matrix[i, j]:.3f}',
                                   ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Cross-modal attention 시각화 저장: {save_path}")
        
        return fig
    
    def visualize_comprehensive_explanation(self, explanation: Dict[str, Any],
                                          image: torch.Tensor,
                                          title: str = "종합 Attention 분석",
                                          save_path: Optional[str] = None) -> plt.Figure:
        """종합적인 설명 시각화"""
        
        fig = plt.figure(figsize=(20, 15), dpi=self.dpi)
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # 그리드 레이아웃 설정
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # 1. 예측 결과 요약
        ax1 = fig.add_subplot(gs[0, 0])
        pred = explanation['prediction']
        
        # 예측 결과 게이지 차트
        confidence = pred['confidence']
        label = pred['label']
        score = pred['score']
        
        theta = np.linspace(0, 2*np.pi, 100)
        r = 0.8
        ax1.fill_between(theta, 0, r * confidence, alpha=0.3, 
                        color='red' if label == 'SPAM' else 'green')
        ax1.set_ylim(0, 1)
        ax1.set_title(f'예측: {label}\\n신뢰도: {confidence:.2f}\\n점수: {score:.3f}')
        ax1.axis('off')
        
        # 2. 텍스트 중요도 (상위 10개)
        ax2 = fig.add_subplot(gs[0, 1:3])
        if explanation['text_analysis']['important_tokens']:
            tokens = explanation['text_analysis']['important_tokens'][:10]
            token_names = [t['token'] for t in tokens]
            importance = [t['combined_importance'] for t in tokens]
            
            bars = ax2.barh(range(len(token_names)), importance, 
                           color=self.text_cmap(np.linspace(0.8, 0.3, len(token_names))))
            ax2.set_yticks(range(len(token_names)))
            ax2.set_yticklabels(token_names)
            ax2.set_title('중요 텍스트 토큰 (Top 10)')
            ax2.set_xlabel('중요도')
        
        # 3. 모달리티 균형
        ax3 = fig.add_subplot(gs[0, 3])
        cross_modal = explanation['cross_modal_analysis']
        text_strength = cross_modal['text_to_image_strength']
        image_strength = cross_modal['image_to_text_strength']
        
        labels = ['Text', 'Image']
        sizes = [text_strength, image_strength]
        colors = ['lightcoral', 'lightskyblue']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax3.set_title('모달리티 기여도')
        
        # 4. 원본 이미지
        ax4 = fig.add_subplot(gs[1, 0])
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0]
            image_np = image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        
        ax4.imshow(image_np)
        ax4.set_title('원본 이미지')
        ax4.axis('off')
        
        # 5. 이미지 attention 히트맵
        ax5 = fig.add_subplot(gs[1, 1])
        if explanation['image_analysis']['important_patches']:
            patches = explanation['image_analysis']['important_patches']
            H, W = image_np.shape[:2]
            patch_size = 16
            patch_h, patch_w = H // patch_size, W // patch_size
            
            attention_map = np.zeros((patch_h, patch_w))
            max_importance = max([p['combined_importance'] for p in patches])
            
            for patch_info in patches:
                coord = patch_info['coordinates']
                importance = patch_info['combined_importance']
                
                if coord[0] < patch_h and coord[1] < patch_w:
                    attention_map[coord[0], coord[1]] = importance / max_importance
            
            attention_resized = cv2.resize(attention_map, (W, H), interpolation=cv2.INTER_LINEAR)
            im = ax5.imshow(attention_resized, cmap=self.image_cmap)
            ax5.set_title('이미지 Attention')
            ax5.axis('off')
            plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        
        # 6. 오버레이
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(image_np)
        if 'attention_resized' in locals():
            ax6.imshow(attention_resized, cmap=self.image_cmap, alpha=0.6)
        ax6.set_title('Attention 오버레이')
        ax6.axis('off')
        
        # 7. 상위 패치 중요도
        ax7 = fig.add_subplot(gs[1, 3])
        if explanation['image_analysis']['important_patches']:
            top_patches = explanation['image_analysis']['important_patches'][:8]
            patch_labels = [f"({p['coordinates'][0]},{p['coordinates'][1]})" for p in top_patches]
            patch_scores = [p['combined_importance'] for p in top_patches]
            
            bars = ax7.bar(range(len(patch_labels)), patch_scores,
                          color=self.image_cmap(np.linspace(0.8, 0.3, len(patch_labels))))
            ax7.set_title('상위 패치 중요도')
            ax7.set_xticks(range(len(patch_labels)))
            ax7.set_xticklabels(patch_labels, rotation=45, fontsize=8)
        
        # 8. 텍스트 하이라이팅
        ax8 = fig.add_subplot(gs[2, :])
        self._visualize_text_highlighting_comprehensive(ax8, explanation)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 종합 분석 시각화 저장: {save_path}")
        
        return fig
    
    def _visualize_text_highlighting_comprehensive(self, ax, explanation: Dict[str, Any]):
        """종합 분석용 텍스트 하이라이팅"""
        
        ax.clear()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 3)
        ax.axis('off')
        ax.set_title('텍스트 중요도 하이라이팅', fontsize=14)
        
        tokens = explanation['text_analysis']['tokens']
        important_tokens = explanation['text_analysis']['important_tokens']
        
        # 토큰별 중요도 매핑
        importance_dict = {item['token']: item['combined_importance'] for item in important_tokens}
        max_importance = max(importance_dict.values()) if importance_dict else 1
        
        x_pos = 0.1
        y_pos = 1.5
        line_height = 0.4
        
        for token in tokens[:80]:  # 처음 80개 토큰
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            display_token = token.replace('##', '')
            if not display_token.strip():
                continue
            
            # 중요도에 따른 색상
            importance = importance_dict.get(token, 0)
            alpha = min(importance / max_importance, 1.0) if max_importance > 0 else 0
            color = self.text_cmap(alpha)
            
            text_width = len(display_token) * 0.12
            
            if x_pos + text_width > 11.5:
                x_pos = 0.1
                y_pos -= line_height
                
            if y_pos < 0.2:
                break
            
            # 배경 박스와 텍스트
            bbox = dict(boxstyle="round,pad=0.02", facecolor=color, alpha=0.8)
            ax.text(x_pos, y_pos, display_token, fontsize=9,
                   bbox=bbox, ha='left', va='center')
            
            x_pos += text_width + 0.08
    
    def create_attention_summary_report(self, explanations: List[Dict[str, Any]],
                                      save_path: str = "attention_summary_report.png"):
        """여러 예측에 대한 attention 요약 보고서 생성"""
        
        if not explanations:
            print("❌ 분석할 예측 결과가 없습니다.")
            return
        
        num_examples = len(explanations)
        fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples), dpi=self.dpi)
        
        if num_examples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('다중 예측 Attention 분석 요약', fontsize=16, fontweight='bold')
        
        for i, explanation in enumerate(explanations):
            pred = explanation['prediction']
            
            # 1. 예측 결과
            axes[i, 0].text(0.5, 0.5, f"예측: {pred['label']}\\n점수: {pred['score']:.3f}\\n신뢰도: {pred['confidence']:.3f}",
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle="round", facecolor='lightblue' if pred['label'] == 'HAM' else 'lightcoral'))
            axes[i, 0].set_xlim(0, 1)
            axes[i, 0].set_ylim(0, 1)
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f'예측 {i+1}')
            
            # 2. 텍스트 중요도
            if explanation['text_analysis']['important_tokens']:
                tokens = explanation['text_analysis']['important_tokens'][:5]
                token_names = [t['token'] for t in tokens]
                importance = [t['combined_importance'] for t in tokens]
                
                bars = axes[i, 1].barh(range(len(token_names)), importance,
                                      color=self.text_cmap(np.linspace(0.8, 0.3, len(token_names))))
                axes[i, 1].set_yticks(range(len(token_names)))
                axes[i, 1].set_yticklabels(token_names)
                axes[i, 1].set_title('Top 5 텍스트 토큰')
            
            # 3. 모달리티 균형
            cross_modal = explanation['cross_modal_analysis']
            text_strength = cross_modal['text_to_image_strength']
            image_strength = cross_modal['image_to_text_strength']
            
            sizes = [text_strength, image_strength]
            colors = ['lightcoral', 'lightskyblue']
            axes[i, 2].pie(sizes, labels=['Text', 'Image'], colors=colors, autopct='%1.1f%%')
            axes[i, 2].set_title('모달리티 균형')
            
            # 4. 상위 패치 중요도
            if explanation['image_analysis']['important_patches']:
                top_patches = explanation['image_analysis']['important_patches'][:5]
                patch_labels = [f"({p['coordinates'][0]},{p['coordinates'][1]})" for p in top_patches]
                patch_scores = [p['combined_importance'] for p in top_patches]
                
                bars = axes[i, 3].bar(range(len(patch_labels)), patch_scores,
                                     color=self.image_cmap(np.linspace(0.8, 0.3, len(patch_labels))))
                axes[i, 3].set_title('Top 5 이미지 패치')
                axes[i, 3].set_xticks(range(len(patch_labels)))
                axes[i, 3].set_xticklabels(patch_labels, rotation=45, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"💾 요약 보고서 저장: {save_path}")
        
        return fig 