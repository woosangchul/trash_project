# 분리해! 인공지능 쓰레기통



개발환경

아나콘다

tensorflow 1.13

numpy

pandas



## 데이터셋 -> TF-record 파일로 변환

모델을 학습시킬 때 학습시킬 이미지 파일을 학습하기 용이하게 TF-record파일로 변환해줘야 한다. TF-record 파일로 변환하면 이미지를 배치처리하기 용이한 장점이 있다



데이터셋은 trainimage폴더 내 Trash 폴더에 저장. 폴더구조는 아래와 같다.

trash 폴더 내 분류할 class label인 can, glass, plastic으로 폴더구조를 생성한다. 폴더 내에 class label에 해당하는 이미지를 넣어준다.



```
trash
  ├─can
  ├─glass
  └─plastic
```



아래 명령어를 통해 이미지를 TF-record 파일로 변환시켜준다. 실행이 완료되면 trainimage 폴더에 TF-Record파일이 생성된다.

```
python download_and_convert_data.py --dataset_name=trash --dataset_dir=trainimage
```



## 모델 학습

모델을 학습하기 전에 학습할 가중치를 불러오기 위해 https://www.dropbox.com/s/k9az7m36gw4orqk/inception_v4.ckpt?dl=0에서 inceptin-v4 ckpt파일을 다운받아 checkpoint_path 폴더에 넣어준다



학습은 아래 명령어를 실행해서 진행한다. 아래 명령어 중에서 checkpoint_exclude_scopes 명령어가 중요한데, 우리는 Custom dataset을 학습시키기 때문에 불러온 모델과 최종 레이어 shape이 일치하지 않는다. 따라서 최종 레이어인 InceptionV4/Logits,InceptionV4/AuxLogits 부분은 복구하지 않고 우리 모델에 맞게 학습시키게 된다.

```
python train_image_classifier.py --train_dir=train_dir 
--dataset_dir=trainimage 
--dataset_name=trash 
--dataset_split_name=train --model_name=inception_v4 
--checkpoint_path=checkpoint_path/inception_v4.ckpt 
--checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits 
--trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits 
--max_number_of_steps=100 
--batch_size=2 --learning_rate=0.01 
--learning_rate_decay_type=fixed 
--save_interval_secs=300 --save_summaries_secs=300 
--log_every_n_steps=200 
--optimizer=rmsprop 
--weight_decay=0.00004 
--clone_on_cpu=True
```



### 그래프 추출 및 서빙모델로 배포

서빙모델로 배포하기전 그래프 구조정보와 체크포인트 파일로 모델을 배포해야 한다. 체크포인트에는 그래프  구조에대한 정보는 없고 노드 이름에 대한 정보와 가중치만 저장되어있기 떄문에 그래프 구조와 체크포인트 파일을 같이 사용해서 freezing 모델로 배포한다

```
python export_inference_graph.py --alsologtostderr --model_name=inception_v4 --output_file=inception_v4_inf_graph_restavailable.pb
```



```
python freeze_graph.py --input_graph=inception_v4_inf_graph_restavailable.pb --input_checkpoint=d:\tmp\train_inception_v4_trash_FineTune_logs\model.ckpt-1000 --input_binary=true --output_graph=inception_v4_graph_rest_available.pb --output_node_names=InceptionV4/Logits/Predictions
```



## TF-Serving을 통해 모델 배포

Builder_serving_model.py 파일에서 아래 부분이 가장 핵심적인 부분.

freeze하여 배포한 모델에 그래프구조와 가중치정보가 함꼐 포함되는데 해당모델을 불러온다. 우리는 RESTFul API를 설계할 때, 바이너리 이미지를 받기로 되어있기때문에 base64로 인코딩한 이미지를 input으로 지정해준다







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



## Docker를 통한 모델 배포

아래 명령어를 통해 Docker server 실행

```
docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=C:/tmp/inceptionv4_serving_restavailable_v5/inceptionV4_v4_1000,target=/models/inceptionV4 -e MODEL_NAME=inceptionV4 -t tensorflow/serving &
```





![image-20210906192525756](C:\Users\woosangchul\AppData\Roaming\Typora\typora-user-images\image-20210906192525756.png)