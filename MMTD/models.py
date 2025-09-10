from transformers import BertForSequenceClassification, BeitForImageClassification, BeitConfig, BertConfig
from transformers.models.bert.modeling_bert import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
import torch
from transformers import CLIPModel, CLIPConfig
from transformers import ViltModel, ViltConfig
import os


class MMTD(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), beit_cfg=BeitConfig(), bert_pretrain_weight=None, beit_pretrain_weight=None, device=None):
        super(MMTD, self).__init__()
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        self.image_encoder = BeitForImageClassification.from_pretrained(beit_pretrain_weight) if beit_pretrain_weight is not None else BeitForImageClassification(beit_cfg)
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh()
        )
        # self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(768, 2)
        self.num_labels = 2
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else "cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        # 입력 텐서들을 self.device로 강제 이동
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)
        text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        text_last_hidden_state = text_outputs.hidden_states[12]
        image_last_hidden_state = image_outputs.hidden_states[12]
        text_last_hidden_state = text_last_hidden_state.to(self.device)
        image_last_hidden_state = image_last_hidden_state.to(self.device)
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size(), device=self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size(), device=self.device)
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = self.pooler(outputs[:, 0, :])
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class BertBeitEmailModelNoCLS(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), beit_cfg=BeitConfig(), bert_pretrain_weight=None, beit_pretrain_weight=None):
        super(BertBeitEmailModelNoCLS, self).__init__()
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        self.image_encoder = BeitForImageClassification.from_pretrained(beit_pretrain_weight) if beit_pretrain_weight is not None else BeitForImageClassification(beit_cfg)
        self.text_encoder.config.output_hidden_states = True
        self.image_encoder.config.output_hidden_states = True
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.squeeze_layer = torch.nn.Linear(768, 1)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(453, 2)
        self.num_labels = 2
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        text_last_hidden_state = text_outputs.hidden_states[12]
        image_last_hidden_state = image_outputs.hidden_states[12]
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size()).to(self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size()).to(self.device)
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = torch.squeeze(self.squeeze_layer(outputs))
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class BertBeitEmailModelFc(torch.nn.Module):
    def __init__(self, bert_cfg=BertConfig(), beit_cfg=BeitConfig(), bert_pretrain_weight=None, beit_pretrain_weight=None):
        super(BertBeitEmailModelFc, self).__init__()
        self.text_encoder = BertForSequenceClassification.from_pretrained(bert_pretrain_weight) if bert_pretrain_weight is not None else BertForSequenceClassification(bert_cfg)
        self.image_encoder = BeitForImageClassification.from_pretrained(beit_pretrain_weight) if beit_pretrain_weight is not None else BeitForImageClassification(beit_cfg)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.Linear(64, 2)
        )
        self.num_labels = 2
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        hidden_state = torch.cat([text_outputs.logits, image_outputs.logits], dim=1)
        logits = self.classifier(hidden_state)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )



class CLIPEmailModel(CLIPModel):
    def __init__(self, config=CLIPConfig()):
        super(CLIPEmailModel, self).__init__(config=config)
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.Tanh()
        )
        self.classifier = torch.nn.Linear(512, 2)
        self.num_labels = 2


    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        out = super(CLIPEmailModel, self).forward(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        text_last_hidden_state = out.text_model_output.last_hidden_state
        text_last_hidden_state = self.text_projection(text_last_hidden_state)
        image_last_hidden_state = out.vision_model_output.last_hidden_state
        image_last_hidden_state512 = self.visual_projection(image_last_hidden_state)
        image_last_hidden_state512 += torch.ones(image_last_hidden_state512.size()).to(self.device)
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state512], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = self.pooler(outputs[:, 0, :])
        logits = self.classifier(outputs)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class ViltEmailModel(ViltModel):
    def __init__(self, config=ViltConfig()):
        super(ViltEmailModel, self).__init__(config=config)
        self.classifier = torch.nn.Linear(768, 2)
        self.num_labels = 2
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None, pixel_mask=None, labels=None):
        out = super(ViltEmailModel, self).forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = self.classifier(out.pooler_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding="same"),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-6, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same"),
            torch.nn.BatchNorm2d(num_features=64, eps=1e-6, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding="same"),
            torch.nn.BatchNorm2d(num_features=64, eps=1e-6, momentum=0.9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(16384, 64),
            torch.nn.BatchNorm1d(num_features=64, eps=1e-6),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.Softmax()
        )
    def forward(self, pixel_values, labels=None):
        out = self.layer1(pixel_values)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.layer4(out)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=200, hidden_size=64, batch_first=True, dropout=0.3)
        self.lstm2 = torch.nn.LSTM(input_size=64, hidden_size=32, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(in_features=32, out_features=2, bias=True)


    def forward(self, input_ids, labels=None):
        out1, _ = self.lstm1(input_ids)
        out2, _ = self.lstm2(out1)
        out = self.fc(out2[:, -1, :])
        logits = torch.nn.functional.softmax(out, dim=1)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class MMA_MF(torch.nn.Module):
    def __init__(self):
        super(MMA_MF, self).__init__()
        self.ltsm = LSTM()
        self.cnn = CNN()
        self.fc = torch.nn.Linear(4, 64)
        self.classifier = torch.nn.Linear(64, 2)

    def forward(self, input_ids, pixel_values, labels=None):
        lstm_out = self.ltsm(input_ids)
        cnn_out = self.cnn(pixel_values)
        lstm_out = lstm_out.logits
        cnn_out = torch.nn.functional.softmax(cnn_out.logits, dim=1)
        out = torch.cat([lstm_out, cnn_out], dim=1)
        out = self.fc(out)
        out = torch.nn.functional.relu(out)
        logits = self.classifier(out)
        logits = torch.nn.functional.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class PretrainedMMTD(torch.nn.Module):
    """사전 훈련된 BERT-BEIT MMTD 모델을 불러오는 클래스"""
    def __init__(self, checkpoint_path="checkpoints/fold5/checkpoint-939/pytorch_model.bin", device=None):
        super(PretrainedMMTD, self).__init__()
        
        # 기존 MMTD 모델 구조로 초기화
        self.mmtd = MMTD(
            device=device
        )
        
        # 사전 훈련된 가중치 로드
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.mmtd.load_state_dict(state_dict)
            print(f"사전 훈련된 모델 로드 완료: {checkpoint_path}")
        else:
            print(f"경고: 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        
        # BERT와 BEIT 인코더를 프리즈
        for param in self.mmtd.text_encoder.parameters():
            param.requires_grad = False
        for param in self.mmtd.image_encoder.parameters():
            param.requires_grad = False
            
        print("BERT와 BEIT 인코더가 프리즈되었습니다.")
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, input_ids, token_type_ids, attention_mask, pixel_values, labels=None):
        return self.mmtd(input_ids, token_type_ids, attention_mask, pixel_values, labels)
    
    def get_model_size(self):
        """모델 크기 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'total_size_mb': total_params * 4 / (1024 * 1024),  # float32 기준
            'trainable_size_mb': trainable_params * 4 / (1024 * 1024)
        }


class HybridMMTDImageTrainable(torch.nn.Module):
    """사전 훈련된 BERT와 새로운 이미지 인코더를 결합한 하이브리드 모델"""
    def __init__(self, pretrained_checkpoint_path="checkpoints/fold5/checkpoint-939/pytorch_model.bin",
                 image_encoder_cls=None, image_pretrain_weight=None, device=None):
        super(HybridMMTDImageTrainable, self).__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 사전 훈련된 BERT 인코더 초기화 (기본 구조로)
        from transformers import BertForSequenceClassification
        self.text_encoder = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
        self.text_encoder.config.output_hidden_states = True
        
        # 사전 훈련된 가중치에서 BERT 부분만 로드
        if os.path.exists(pretrained_checkpoint_path):
            try:
                state_dict = torch.load(pretrained_checkpoint_path, map_location='cpu')
                
                # BERT 관련 가중치만 추출 (classifier 제외)
                bert_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("text_encoder."):
                        # 키에서 "text_encoder." 제거
                        new_key = key.replace("text_encoder.", "")
                        # classifier 레이어는 제외 (크기 불일치 방지)
                        if not new_key.startswith("classifier."):
                            bert_state_dict[new_key] = value
                
                # BERT 모델에 가중치 로드 (strict=False로 설정하여 일부 키가 없어도 무시)
                missing_keys, unexpected_keys = self.text_encoder.load_state_dict(bert_state_dict, strict=False)
                print(f"사전 훈련된 BERT 가중치 로드 완료 (classifier 제외): {pretrained_checkpoint_path}")
                if missing_keys:
                    print(f"누락된 키 수: {len(missing_keys)} (classifier 포함)")
                if unexpected_keys:
                    print(f"예상치 못한 키 수: {len(unexpected_keys)}")
                    
            except Exception as e:
                print(f"체크포인트 로딩 중 오류 발생: {e}")
                print("기본 사전 훈련된 BERT 모델을 사용합니다.")
        else:
            print(f"경고: 체크포인트 파일을 찾을 수 없습니다: {pretrained_checkpoint_path}")
            print("기본 사전 훈련된 BERT 모델을 사용합니다.")
        
        # BERT 인코더를 프리즈
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # 새로운 이미지 인코더 초기화
        if image_encoder_cls and image_pretrain_weight:
            self.image_encoder = image_encoder_cls.from_pretrained(image_pretrain_weight)
            if hasattr(self.image_encoder, 'config'):
                self.image_encoder.config.output_hidden_states = True
        else:
            raise ValueError("image_encoder_cls와 image_pretrain_weight를 제공해야 합니다.")
        
        # 멀티모달 레이어들
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh()
        )
        self.classifier = torch.nn.Linear(768, 2)
        self.num_labels = 2
        
        print("BERT 인코더가 프리즈되었습니다. 이미지 인코더만 학습됩니다.")
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None, token_type_ids=None):
        # 입력 텐서들을 디바이스로 이동
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # 텍스트 인코딩 (프리즈된 BERT)
        text_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        # token_type_ids가 있고 모델이 지원하는 경우에만 추가
        if token_type_ids is not None and hasattr(self.text_encoder, 'bert'):
            text_kwargs['token_type_ids'] = token_type_ids
        
        with torch.no_grad():
            text_outputs = self.text_encoder(**text_kwargs)
        
        # 이미지 인코딩 (학습 가능)
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        
        # 텍스트 hidden states 추출
        if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states:
            text_last_hidden_state = text_outputs.hidden_states[-1]
        else:
            # hidden_states가 없는 경우 처리
            if hasattr(text_outputs, 'pooler_output'):
                text_last_hidden_state = text_outputs.pooler_output.unsqueeze(1)
                # 기본 시퀀스 길이 설정
                text_last_hidden_state = text_last_hidden_state.repeat(1, 197, 1)  # BEiT 기본값
            elif hasattr(text_outputs, 'last_hidden_state'):
                text_last_hidden_state = text_outputs.last_hidden_state
            else:
                # 마지막 수단
                batch_size = input_ids.shape[0]
                text_last_hidden_state = torch.zeros(batch_size, 197, 768, device=self.device)
        
        # 텍스트 projection 적용 (필요한 경우)
        if hasattr(self, 'text_projection') and self.text_projection is not None:
            text_last_hidden_state = self.text_projection(text_last_hidden_state)
        
        # 이미지 출력의 hidden_states 추출 및 차원 처리
        if hasattr(image_outputs, 'hidden_states') and image_outputs.hidden_states:
            image_last_hidden_state = image_outputs.hidden_states[-1]
            # MobileViT 등은 4차원 출력 (batch, height, width, channels)이 나올 수 있음
            if len(image_last_hidden_state.shape) == 4:
                # 4차원을 3차원으로 변환: (batch, H*W, channels)
                b, h, w, c = image_last_hidden_state.shape
                image_last_hidden_state = image_last_hidden_state.view(b, h*w, c)
                # 768 차원으로 맞춤
                if c != 768:
                    if not hasattr(self, 'image_projection'):
                        self.image_projection = torch.nn.Linear(c, 768).to(self.device)
                    image_last_hidden_state = self.image_projection(image_last_hidden_state)
            elif len(image_last_hidden_state.shape) == 3:
                # 3차원 출력 (batch, seq_len, hidden_size) - ViT-Tiny 등
                c = image_last_hidden_state.shape[-1]
                if c != 768:
                    if not hasattr(self, 'image_projection'):
                        self.image_projection = torch.nn.Linear(c, 768).to(self.device)
                    image_last_hidden_state = self.image_projection(image_last_hidden_state)
        elif hasattr(image_outputs, 'last_hidden_state'):
            image_last_hidden_state = image_outputs.last_hidden_state
            # 차원 확인 및 처리
            if len(image_last_hidden_state.shape) == 4:
                b, h, w, c = image_last_hidden_state.shape
                image_last_hidden_state = image_last_hidden_state.view(b, h*w, c)
                if c != 768:
                    if not hasattr(self, 'image_projection'):
                        self.image_projection = torch.nn.Linear(c, 768).to(self.device)
                    image_last_hidden_state = self.image_projection(image_last_hidden_state)
            elif len(image_last_hidden_state.shape) == 3:
                # 3차원 출력 - ViT-Tiny 등
                c = image_last_hidden_state.shape[-1]
                if c != 768:
                    if not hasattr(self, 'image_projection'):
                        self.image_projection = torch.nn.Linear(c, 768).to(self.device)
                    image_last_hidden_state = self.image_projection(image_last_hidden_state)
        elif hasattr(image_outputs, 'pooler_output'):
            # pooler_output이 있는 경우 (보통 1차원)
            image_last_hidden_state = image_outputs.pooler_output.unsqueeze(1)
            # 텍스트와 시퀀스 길이 맞춤
            text_seq_len = text_last_hidden_state.shape[1]
            image_last_hidden_state = image_last_hidden_state.repeat(1, text_seq_len, 1)
            # 768 차원으로 맞춤
            if image_last_hidden_state.shape[-1] != 768:
                if not hasattr(self, 'image_projection'):
                    self.image_projection = torch.nn.Linear(image_last_hidden_state.shape[-1], 768).to(self.device)
                image_last_hidden_state = self.image_projection(image_last_hidden_state)
        else:
            # 마지막 수단: 제로 텐서 생성
            batch_size = pixel_values.shape[0]
            text_seq_len = text_last_hidden_state.shape[1]
            image_last_hidden_state = torch.zeros(batch_size, text_seq_len, 768, device=self.device)
        
        # 차원 맞춤
        text_last_hidden_state = text_last_hidden_state.to(self.device)
        image_last_hidden_state = image_last_hidden_state.to(self.device)
        
        # 위치 임베딩 추가
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size(), device=self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size(), device=self.device)
        
        # 멀티모달 융합
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = self.pooler(outputs[:, 0, :])
        logits = self.classifier(outputs)
        
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def get_model_size(self):
        """모델 크기 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'total_size_mb': total_params * 4 / (1024 * 1024),
            'trainable_size_mb': trainable_params * 4 / (1024 * 1024)
        }

class HybridMMTDTextTrainable(torch.nn.Module):
    """사전 훈련된 BEiT와 새로운 텍스트 인코더를 결합한 하이브리드 모델 (텍스트 인코더만 학습)"""
    def __init__(self, pretrained_checkpoint_path="checkpoints/fold5/checkpoint-939/pytorch_model.bin",
                 text_encoder_cls=None, text_pretrain_weight=None, device=None):
        super(HybridMMTDTextTrainable, self).__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 새로운 텍스트 인코더 초기화
        if text_encoder_cls and text_pretrain_weight:
            self.text_encoder = text_encoder_cls.from_pretrained(text_pretrain_weight)
            if hasattr(self.text_encoder, 'config'):
                self.text_encoder.config.output_hidden_states = True
        else:
            raise ValueError("text_encoder_cls와 text_pretrain_weight를 제공해야 합니다.")
        
        # 사전 훈련된 BEiT 인코더 초기화 (기본 구조로)
        from transformers import BeitForImageClassification
        self.image_encoder = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
        self.image_encoder.config.output_hidden_states = True
        
        # 사전 훈련된 가중치에서 BEiT 부분만 로드
        if os.path.exists(pretrained_checkpoint_path):
            try:
                state_dict = torch.load(pretrained_checkpoint_path, map_location='cpu')
                
                # BEiT 관련 가중치만 추출 (classifier 제외)
                beit_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("image_encoder."):
                        # 키에서 "image_encoder." 제거
                        new_key = key.replace("image_encoder.", "")
                        # classifier 레이어는 제외 (크기 불일치 방지)
                        if not new_key.startswith("classifier."):
                            beit_state_dict[new_key] = value
                
                # BEiT 모델에 가중치 로드 (strict=False로 설정하여 일부 키가 없어도 무시)
                missing_keys, unexpected_keys = self.image_encoder.load_state_dict(beit_state_dict, strict=False)
                print(f"사전 훈련된 BEiT 가중치 로드 완료 (classifier 제외): {pretrained_checkpoint_path}")
                if missing_keys:
                    print(f"누락된 키 수: {len(missing_keys)} (classifier 포함)")
                if unexpected_keys:
                    print(f"예상치 못한 키 수: {len(unexpected_keys)}")
                    
            except Exception as e:
                print(f"체크포인트 로딩 중 오류 발생: {e}")
                print("기본 사전 훈련된 BEiT 모델을 사용합니다.")
        else:
            print(f"경고: 체크포인트 파일을 찾을 수 없습니다: {pretrained_checkpoint_path}")
            print("기본 사전 훈련된 BEiT 모델을 사용합니다.")
        
        # # BEiT 인코더를 프리즈
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False
        
        # 멀티모달 레이어들
        self.multi_modality_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.Tanh()
        )
        self.classifier = torch.nn.Linear(768, 2)
        self.num_labels = 2
        
        # 텍스트 인코더의 hidden_size를 768로 맞추기 위한 projection layer
        # MobileBERT는 512, DistilBERT는 768, TinyBERT는 312 등 다양할 수 있음
        if hasattr(self.text_encoder.config, 'hidden_size'):
            text_hidden_size = self.text_encoder.config.hidden_size
            if text_hidden_size != 768:
                self.text_projection = torch.nn.Linear(text_hidden_size, 768)
                print(f"텍스트 projection layer 추가: {text_hidden_size} -> 768")
            else:
                self.text_projection = None
        else:
            self.text_projection = None
        
        print("BEiT 인코더가 프리즈되었습니다. 텍스트 인코더만 학습됩니다.")
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None, token_type_ids=None):
        # 입력 텐서들을 디바이스로 이동
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        
        # 텍스트 인코딩 (학습 가능)
        text_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        # token_type_ids가 있고 모델이 지원하는 경우에만 추가
        if token_type_ids is not None and hasattr(self.text_encoder, 'bert'):
            text_kwargs['token_type_ids'] = token_type_ids
        
        text_outputs = self.text_encoder(**text_kwargs)
        
        # 이미지 인코딩 (프리즈된 BEiT)
        with torch.no_grad():
            image_outputs = self.image_encoder(pixel_values=pixel_values)
        
        # 텍스트 hidden states 추출
        if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states:
            text_last_hidden_state = text_outputs.hidden_states[-1]
        else:
            # hidden_states가 없는 경우 처리
            if hasattr(text_outputs, 'pooler_output'):
                text_last_hidden_state = text_outputs.pooler_output.unsqueeze(1)
                # 기본 시퀀스 길이 설정
                text_last_hidden_state = text_last_hidden_state.repeat(1, 197, 1)  # BEiT 기본값
            elif hasattr(text_outputs, 'last_hidden_state'):
                text_last_hidden_state = text_outputs.last_hidden_state
            else:
                # 마지막 수단
                batch_size = input_ids.shape[0]
                text_last_hidden_state = torch.zeros(batch_size, 197, 768, device=self.device)
        
        # 텍스트 projection 적용 (필요한 경우)
        if hasattr(self, 'text_projection') and self.text_projection is not None:
            text_last_hidden_state = self.text_projection(text_last_hidden_state)
        
        # 이미지 출력의 hidden_states 추출 및 차원 처리
        if hasattr(image_outputs, 'hidden_states') and image_outputs.hidden_states:
            image_last_hidden_state = image_outputs.hidden_states[-1]
        else:
            # fallback: last_hidden_state 사용
            if hasattr(image_outputs, 'last_hidden_state'):
                image_last_hidden_state = image_outputs.last_hidden_state
            else:
                batch_size = pixel_values.shape[0]
                image_last_hidden_state = torch.zeros(batch_size, 197, 768, device=self.device)
        
        # 차원 맞춤
        text_last_hidden_state = text_last_hidden_state.to(self.device)
        image_last_hidden_state = image_last_hidden_state.to(self.device)
        
        # 위치 임베딩 추가
        text_last_hidden_state += torch.zeros(text_last_hidden_state.size(), device=self.device)
        image_last_hidden_state += torch.ones(image_last_hidden_state.size(), device=self.device)
        
        # 멀티모달 융합
        fuse_hidden_state = torch.cat([text_last_hidden_state, image_last_hidden_state], dim=1)
        outputs = self.multi_modality_transformer_layer(fuse_hidden_state)
        outputs = self.pooler(outputs[:, 0, :])
        logits = self.classifier(outputs)
        
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def get_model_size(self):
        """모델 크기 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'total_size_mb': total_params * 4 / (1024 * 1024),
            'trainable_size_mb': trainable_params * 4 / (1024 * 1024)
        }
