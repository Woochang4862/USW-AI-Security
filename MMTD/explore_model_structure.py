import torch
import sys
sys.path.append('.')

from models import MMTD

def explore_model_structure():
    """실제 MMTD 모델 구조 탐색"""
    print("🔍 MMTD 모델 구조 탐색 시작")
    print("="*60)
    
    # 모델 생성
    model = MMTD(
        bert_pretrain_weight='bert-base-multilingual-cased',
        beit_pretrain_weight='microsoft/dit-base'
    )
    
    print(f"📊 전체 모델 구조:")
    print(f"   - Text Encoder: {type(model.text_encoder).__name__}")
    print(f"   - Image Encoder: {type(model.image_encoder).__name__}")
    print(f"   - Fusion Layer: {type(model.multi_modality_transformer_layer).__name__}")
    
    print(f"\n🔤 BERT Text Encoder 구조 탐색:")
    print(f"   Type: {type(model.text_encoder.bert)}")
    
    # BERT encoder layers 확인
    if hasattr(model.text_encoder.bert, 'encoder'):
        encoder = model.text_encoder.bert.encoder
        print(f"   Encoder layers: {len(encoder.layer)}개")
        
        # 첫 번째 레이어 구조 확인
        first_layer = encoder.layer[0]
        print(f"   First layer type: {type(first_layer)}")
        print(f"   First layer attributes: {list(first_layer.__dict__.keys())}")
        
        if hasattr(first_layer, 'attention'):
            attention = first_layer.attention
            print(f"   Attention type: {type(attention)}")
            print(f"   Attention attributes: {list(attention.__dict__.keys())}")
            
            if hasattr(attention, 'self'):
                self_attn = attention.self
                print(f"   Self-attention type: {type(self_attn)}")
                print(f"   Self-attention attributes: {list(self_attn.__dict__.keys())}")
    
    print(f"\n🖼️ BEiT Image Encoder 구조 탐색:")
    print(f"   Type: {type(model.image_encoder.beit)}")
    
    # BEiT encoder layers 확인
    if hasattr(model.image_encoder.beit, 'encoder'):
        encoder = model.image_encoder.beit.encoder
        print(f"   Encoder layers: {len(encoder.layer)}개")
        
        # 첫 번째 레이어 구조 확인
        first_layer = encoder.layer[0]
        print(f"   First layer type: {type(first_layer)}")
        print(f"   First layer attributes: {list(first_layer.__dict__.keys())}")
        
        if hasattr(first_layer, 'attention'):
            attention = first_layer.attention
            print(f"   Attention type: {type(attention)}")
            print(f"   Attention attributes: {list(attention.__dict__.keys())}")
            
            if hasattr(attention, 'attention'):
                self_attn = attention.attention
                print(f"   Self-attention type: {type(self_attn)}")
                print(f"   Self-attention attributes: {list(self_attn.__dict__.keys())}")
    
    print(f"\n🔗 Fusion Layer 구조:")
    fusion = model.multi_modality_transformer_layer
    print(f"   Type: {type(fusion)}")
    print(f"   Attributes: {list(fusion.__dict__.keys())}")
    
    if hasattr(fusion, 'self_attn'):
        print(f"   Self-attention type: {type(fusion.self_attn)}")
        print(f"   Self-attention attributes: {list(fusion.self_attn.__dict__.keys())}")
    
    # 작은 테스트 실행하여 실제 output 구조 확인
    print(f"\n🧪 실제 Forward Pass 테스트:")
    model.eval()
    
    with torch.no_grad():
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10),
            'token_type_ids': torch.zeros(1, 10),
            'pixel_values': torch.randn(1, 3, 224, 224)
        }
        
        # 텍스트 인코더만 실행
        print("   텍스트 인코더 출력 구조:")
        text_outputs = model.text_encoder(**{k: v for k, v in dummy_input.items() if k != 'pixel_values'})
        print(f"     Type: {type(text_outputs)}")
        print(f"     Attributes: {list(text_outputs.__dict__.keys()) if hasattr(text_outputs, '__dict__') else 'No dict'}")
        if hasattr(text_outputs, 'hidden_states'):
            print(f"     Hidden states: {len(text_outputs.hidden_states)} layers")
        if hasattr(text_outputs, 'attentions') and text_outputs.attentions is not None:
            print(f"     Attentions: {len(text_outputs.attentions)} layers")
        
        # 이미지 인코더만 실행
        print("   이미지 인코더 출력 구조:")
        image_outputs = model.image_encoder(pixel_values=dummy_input['pixel_values'])
        print(f"     Type: {type(image_outputs)}")
        print(f"     Attributes: {list(image_outputs.__dict__.keys()) if hasattr(image_outputs, '__dict__') else 'No dict'}")
        if hasattr(image_outputs, 'hidden_states'):
            print(f"     Hidden states: {len(image_outputs.hidden_states)} layers")
        if hasattr(image_outputs, 'attentions') and image_outputs.attentions is not None:
            print(f"     Attentions: {len(image_outputs.attentions)} layers")
    
    print(f"\n✅ 모델 구조 탐색 완료!")

if __name__ == "__main__":
    explore_model_structure() 