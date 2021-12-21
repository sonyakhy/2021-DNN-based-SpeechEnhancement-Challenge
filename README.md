# 2021 AI Challenge   
DNN-based Speech Enhancement in the frequency domain  

# 팀 소개
팀명 : 믿거조  
팀장 : 차재빈  
팀원 : 허명범, 이연재, 김승연, 서은빈  
주제 : Data Augmentation을 통한 DNN 기반 음성 향상

# 목차
1. 연구 소개
2. 실험 환경
3. 모델 별 성능 측정
4. 네트워크 모델 구조 변형
5. 다양한 데이터 증강 기법
6. 데이터 증강을 통한 성능 측정
7. 결론

# 1. 연구 소개
> Speech Enhancement란?  
* 음성향상기술은 잡음이 섞인 음성데이터에서 잡음 음성만을 분리하는 기술이다.  
최근 딥러닝을 이용한 음성 향상 기술은 높은 성과를 보이고 있다.  

* 응용 분야로는 음성인식, 화상회의, 보청기 등 많은 곳에 쓰인다.  
> 연구 목적
1. 주어진 음성 데이터에 대한 최적의 모델 탐색
2. 모델 구조 변형을 통한 성능 비교
3. 데이터 증강을 통한 성능 향상

# 2. 실험 환경
> **Setting**  
* CPU : i7-11700 2.50GHz  
* GPU : GeForce GTX 1080 Ti  
* RAM : 16GB   
* Framework : Pytorch  
* Optimizer : Adam 
* Network : DCCRN 
* Loss function : SI-SNR  
* Learning rate : 0.001  
* Batch size : 10  
* SNR(dB) : [5, 10, 15]  
* Learning method : T-F masking  
   

> **Dataset**  
* DNS Challenge Dataset  
Clean Data :  1680개    
Noise Data : 298개  

* Noisy Data :   
Clean과 Noise 데이터의 Train/Validation 비율을 3:1로 고정  
즉, Train : Validation = 1120 : 560  
SNR 5,10,15dB 환경으로 Noisy 데이터 생성 (Noise를 입힌 Clean 데이터) : 3360개  
Augmentation (Data Shifting, Minus, Reverse)을 통한 Noisy 데이터 생성 : 각 기법 당 3360개   
**총 Noisy Data : 13440개**  
* Deep Noise Suppression (DNS) Challenge 4 - ICASSP 2022 [[link]](https://github.com/microsoft/DNS-Challenge)  

> **Evaluation metrics**
* PESQ (Perceptual Evaluation of Speech Quality)  


# 3. 모델 별 성능 측정 및 비교  
모든 실험은 Loss가 적절히 수렴했다고 판단되었음을 가정한다.  
  
> Network에 따른 성능 비교  
* DCCRN (Blue)
* FullSubNet (Orange)
![image](https://user-images.githubusercontent.com/87358781/146786508-cb499d8d-9ec0-4d60-a64a-93fad7f53d09.png)  
Noisy Data 3360개를 동일한 실험 조건으로 Network만 변경하여 실험한다.  
결과적으로 DCCRN에서는 PESQ가 2.140, FullSubNet에서는 2.112으로 DCCRN인 경우 더 나은 성능을 보인다.  
  
> Loss에 따른 성능 비교
* MSE, SI-SNR, SDR, SI-SDR  
|제목|내용|설명|  
|:---|---:|:---:|  
|왼쪽정렬|오른쪽정렬|중앙정렬|  
|왼쪽정렬|오른쪽정렬|중앙정렬|  
|왼쪽정렬|오른쪽정렬|중앙정렬|  

Noisy Data 1120개, Network 모델은 DCCRN을 기준으로 손실함수만 변경하여 실험한다.  
결과적으로 손실함수(Loss)가 SI-SNR인 경우 가장 좋은 성능을 보인다.  

> Perceptual loss 추가에 따른 성능 비교
* 추가하지 않음 (Blue) 
* LMS 추가 (Gray)
* PMSQE 추가 (Red)  
Perceptual loss는 지각적 성능 향상을 목적으로 한다.  
![image](https://user-images.githubusercontent.com/87358781/146793078-2490b53f-2791-417c-8057-0f9b337bb94f.png)    
Data 3360개, DCCRN, SI-SNR을 기준으로 Perceptual Loss를 추가하였을 때,  
PESQ는 PMSQE일 때 2.162, LMS는 2.153으로 증가함을 볼 수 있으며, PMSQE에서 가장 큰 상승을 보인다.  
더불어, PMSQ 수치 뿐 아니라 사람이 실제로 느끼는 음성의 질이 매우 좋아짐을 볼 수 있었다.  

# 4. 네트워크 모델 구조 변형
![CRN Structure](https://user-images.githubusercontent.com/87358781/146889875-397d6e1f-4fef-4751-856f-fa7113c7f2d8.png)  
1. LSTM 구조 변형 
> LSTM 개수 변화에 따른 성능 비교  
* 기존 (2개)
* 기존 -1 (1개)
* 기존 +1 (3개)
* 기존 +2 (4개)  
DCCRN의 구조를 보면 LSTM이 2개임을 볼 수 있다. LSTM의 수를 변화시켰을 때 성능 비교는 다음과 같다.  
![image](https://user-images.githubusercontent.com/87358781/146795418-00951943-4ada-49a7-8c6a-5eab0aecb33a.png)  
Data 3360개, DCCRN, SI-SNR, PMSQE를 적용시켰을 때, 성능 차이는 미미했고, 기존 모델인 LSTM이 2개일 때 가장 좋은 성능을 보인다.  
1. Convolution Layer 구조 변형
> Convolution Layer 개수 변화에 따른 성능 비교
* 기존 (6층)
* 기존 -1 (5층)
* 기존 +1 (7층)
* 기존 +2 (8층)  

# 5. 데이터 증강 기법을 통한 성능 향상
> Data Augmentation이란?
* 원본 데이터에 인위적인 변화를 주어 기존과는 다른 데이터를 형성하는 법을 말한다.  
* 인간에게는 같은 데이터로 보일 수 있지만 인공지능에게는 매우 다른 데이터가 되어 학습에 도움을 줄 수 있다.  

1. Shifting the sound  
* 음성 데이터의 시퀀스를 shift하는 방법
* 순서가 바뀌어 들리게 된다.
기존 데이터 3360개에 Shifting한 데이터 3360개를 더한 7720개 데이터셋 생성  
![image](https://user-images.githubusercontent.com/87358781/146796910-d8d009aa-24af-4bc9-bd7c-db7f20b7038f.png)  
결과적으로 데이터 증강 전에는 PESQ가 2.238, 증강 후에는 2.490로 대폭 향상됨을 볼 수 있다.

2. Reverse the sound
* 음성 데이터를 거꾸로 변환하는 방법
* 음성이 거꾸로 들리게 된다.  
기존 데이터 3360개에 Reverse한 데이터 3360개를 더한 7720개 데이터셋 생성  
![image](https://user-images.githubusercontent.com/87358781/146798422-d40fba4f-a365-4692-a4db-6b2a742b0bb9.png)  
결과적으로 Shifting보다는 저조했지만, PESQ가 2.238에서 2.462로 대폭 향상됨을 볼 수 있다.  

3. Minus the sound 
* 음성 데이터의 위상을 뒤집는 방법
* 사람이 듣기에는 원본 데이터와 동일하게 들린다.  
기존 데이터 3360개에 Reverse한 데이터 3360개를 더한 7720개 데이터셋 생성
![image](https://user-images.githubusercontent.com/87358781/146798834-efc600ac-53bb-4627-bc71-82bc6d1ea9c4.png)  
결과적으로 Shifting보다는 저조했지만, PESQ가 2.238에서 2.451로 대폭 향상됨을 볼 수 있다.  
* **세 가지 데이터 증강 기법을 비교하면 Shifting, Minus, Reverse 순서로 성능이 향상된다.**  

# 7. 결론
> 전체 Augmentation 데이터를 통한 성능 향상 비교 
* 원본 데이터 3360개, Shifting/Reverse/Minus를 통한 증강 데이터 10080개 데이터를 결합  
 (부족한 메모리로 인해 결합에 어려움을 겪었지만 Swap Memory를 증가시켜 해결함)
* 총 13440개 데이터셋으로 실험  



