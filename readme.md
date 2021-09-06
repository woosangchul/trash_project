# 분리해! 인공지능 쓰레기통



#### 개발환경

- 아나콘다

- tensorflow 1.13, numpy, pandas, matplotlib, jupyter



#### **사전준비**

- 우리 프로젝트는 fine-tune 기법을 사용하므로 사전훈련된 inception-v4 가중치가 필요하다. [inception-v4 체크포인트 파일](https://www.dropbox.com/s/k9az7m36gw4orqk/inception_v4.ckpt?dl=0)을 다운받아 train/1_inception_v4_checkpoint 폴더에 넣어준다.



## 1. 이미지 -> TF-record 파일로 변환

- 모델을 학습시키기 위해서는 TF-record파일이 필요하다. 학습할 이미지를 TF-record파일로 변환하기 위해 download_and_convert_data.py를 사용한다. TF-record 파일로 변환하면 이미지를 배치처리하기 용이한 장점이 있다.

- 데이터셋은 train폴더 내 2_train_image\trash 폴더에 저장돼어 있으며, 폴더구조는 아래와 같다.
  - trash 폴더 내 분류할 class label인 can, glass, plastic으로 폴더구조를 생성한다. 폴더 내에 class label에 해당하는 이미지를 넣어준다.

```
train  
├─2_train_image
│  └─trash
│      ├─can
│      ├─glass
│      └─plastic
```



- 다른 Custom Dataset을 활용하고 싶다면 dataset/trash.py 파일에서 아래 2개 값을 수정한다.

```
<trash.py>

34 line : SPLITS_TO_SIZES = {'train': 15, 'validation': 5}
36 line : _NUM_CLASSES = 3
```



- 아래 명령어를 통해 이미지를 TF-record 파일로 변환시켜준다. 실행이 완료되면 train\2_train_image 폴더에 TF-Record파일이 생성된다.

```
python download_and_convert_data.py --dataset_name=trash --dataset_dir=train\2_train_image
```



## 2. 모델 학습

- 학습은 아래 명령어를 실행해서 진행한다. 아래 명령어 중에서 checkpoint_exclude_scopes 명령어가 중요한데, 불러온 inception_v4 체크포인트 파일은 1001개 class label을 가지는 imagenet 데이터셋을 학습한 모델이다. 우리 프로젝트에서 학습할 데이터셋은 3개의 class label만을 가지므로  불러온 모델과 최종 레이어 shape이 일치하지 않는다. 따라서 최종 레이어인 InceptionV4/Logits,InceptionV4/AuxLogits 부분은 불러오지 않고 해당 노드만 학습시키게 된다.
- --clone_on_cpu=True 명령어는 cpu를 사용할때만 넣어주고 gpu를 사용할때는 삭제시킨다.
- 배치사이즈는 batch_size명령어를 통해 조절가능하고, 학습 횟수는 max_number_of_steps 명령어를 통해 가능하다.

```
python train_image_classifier.py --train_dir=train/3_checkpoint_dir --dataset_dir=train/2_train_image --dataset_name=trash --dataset_split_name=train --model_name=inception_v4 --checkpoint_path=train/1_inception_v4_checkpoint/inception_v4.ckpt --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits --max_number_of_steps=100 --batch_size=2 --learning_rate=0.01 --learning_rate_decay_type=fixed --save_interval_secs=300 --save_summaries_secs=300 --log_every_n_steps=200 --optimizer=rmsprop --weight_decay=0.00004 --clone_on_cpu=True
```



## 3. 그래프 추출 및 서빙모델로 배포

- TF-Serving을 사용하기전, 그래프를 freeze 시킨 파일을 얻어야 한다. checkpoint파일에는 학습에 필요한 모든 가중치들이 저장되어있으므로 freeze시켜 불필요한 정보들을 제거해야 한다. 또 학습할때와 예측할떄의 그래프 구조가 조금 달리지므로 freeze 시켜야 한다.
- 체크포인트파일을 통해 그래프를 freeze 시키기 전, 그래프 구조를 추출해야 한다. 체크포인트 파일은 노드에대한 이름과 가중치에 대한 정보만 담고있으므로 그래프 구조를 모르면 사용할 수 없다 따라서 freeze시키기 위해서는 그래프 정보를 담고있는 파일과 체크포인트 파일이 필요하다
- 아래 명령어를 통해 그래프 구조를 추출한다.

```
python export_inference_graph.py --alsologtostderr --model_name=inception_v4 --output_file=inception_v4_inf_graph_restavailable.pb
```

- 추출된 그래프정보와 학습한 체크포인트 파일을 사용하여 그래프구조와 파라미터 정보를 담을 수 있는 pb파일로 추출하게 된다.

```
python freeze_graph.py --input_graph=inception_v4_inf_graph_restavailable.pb --input_checkpoint=d:\tmp\train_inception_v4_trash_FineTune_logs\model.ckpt-1000 --input_binary=true --output_graph=inception_v4_graph_rest_available.pb --output_node_names=InceptionV4/Logits/Predictions
```



## 4. TF-Serving을 통해 모델 배포

- Builder_serving_model.py 파일에서 아래 부분이 가장 핵심적인 부분.

- 우리는 RESTFul API를 사용할 때, input에 바이너리 이미지를 바로 전달할 수 없어 base64로 인코딩한 이미지를 전달한다. 따라서 우리가 학습시킨 모델과 input이 달리진다. Serving을 통해 모델을 배포하기전에 base64이미지르 input으로 하고 바이너리로 전처리하는 노드를 추가하여 모델을 배포한다.
- input으로 사용할 텐서이름을 추출하여 inputs 변수에 딕셔너리 형태로 넣어준다. output도 마찬가지로 진행한다.



```
with tf.Graph().as_default() as g:
    with tf.Session() as new_sess:
        with tf.gfile.FastGFile("inception_v4_graph_rest_available.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        
        input_placeholder = new_sess.graph.get_tensor_by_name("image_preprocessing/input_bytes:0")
        softmax = new_sess.graph.get_tensor_by_name("InceptionV4/Logits/Predictions:0")
        processed_image = new_sess.graph.get_tensor_by_name("image_preprocessing/Reshape_1:0")
        
        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_placeholder)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(softmax)
        tensor_info_image = tf.saved_model.utils.build_tensor_info(processed_image)
        prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y,
                'images' : tensor_info_image
                },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

```



## 5. Docker를 통한 모델 배포

- [도커 설치방법](https://github.com/woosangchul/serving/blob/master/tensorflow_serving/g3doc/docker.md) 링크를 통해 도커를 설치한다.

- 아래 명령어를 통해 Docker server 실행. source 부분만 4번에서 배포한 경로로 설정해주면 된다.

```
docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=C:/tmp/inceptionv4_serving_restavailable_v5/inceptionV4_v4_1000,target=/models/inceptionV4 -e MODEL_NAME=inceptionV4 -t tensorflow/serving &
```

- Docker서버 작동화면

![도커서버 구동화면](z_readme image\docker.png)



- 주피터노트북에서 실행한 화면

![주피터노트북 실행화면](z_readme image\jupyter.png)

