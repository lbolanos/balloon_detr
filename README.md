"# balloon_detr" 

https://towardsdatascience.com/installing-cvat-intels-computer-vision-annotation-tool-on-the-cloud-c7759ae28f0e
http://localhost:8080/auth/login
http://localhost:8080/admin

sudo sh -c 'export CVAT_HOST=localhost && docker-compose -f docker-compose.yml -f docker-compose.override.yml -f components/serverless/docker-compose.serverless.yml up -d --build'
sudo docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'

docker ps -a

wget https://github.com/nuclio/nuclio/releases/download/1.5.16/nuctl-1.5.16-linux-amd64
sudo chmod +x nuctl-1.5.16-linux-amd64

sudo ln -sf $(pwd)/nuctl-1.5.16-linux-amd64 /usr/local/bin/nuctl


nuctl create project cvat

nuctl deploy --project-name cvat \
  --path serverless/openvino/dextr/nuclio \
  --volume `pwd`/serverless/common:/opt/nuclio/common \
  --platform local

nuctl deploy --project-name cvat \
  --path serverless/openvino/omz/public/yolo-v3-tf/nuclio \
  --volume `pwd`/serverless/common:/opt/nuclio/common \
  --platform local

nuctl get function --namespace nuclio

image=$(curl https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png --output - | base64 | tr -d '\n')
cat << EOF > /tmp/input.json
{"image": "$image"}
EOF
cat /tmp/input.json | nuctl invoke openvino-omz-public-yolo-v3-tf -c 'application/json'



s3fs dincro-cvat ~/s3-bucket -o allow_other -o passwd_file=/home/ubuntu/.passwd-s3fs -o url=https://s3.amazonaws.com -o use_path_request_style -o endpoint=fr-par -o parallel_count=15 -o multipart_size=128 -o nocopyap
sudo s3fs dincro-cvat /mnt/s3-cvat -o allow_other -o passwd_file=/home/ubuntu/.passwd-s3fs -o nonempty -o use_path_request_style -o endpoint=fr-par -o parallel_count=15 -o multipart_size=128 -o nocopyapi  -o gid=1001 -o mp_umask=002
sudo s3fs dincro-cvat /mnt/s3-cvat -o passwd_file=${HOME}/.passwd-s3fs


nuctl deploy --project-name cvat \
  --path serverless/balloons \
  --platform local

nuctl get function --namespace nuclio
http://localhost:8070/projects/cvat/functions
docker ps -a
docker logs 845aba923042 > balloons.log

image=$(curl https://www.thepartycompany.co.uk/image/cache/data/new%20balloons/flame-red-party-balloon-800x800.jpg --output - | base64 | tr -d '\n')
cat << EOF > /tmp/input.json
{"image": "$image"}
EOF
cat /tmp/input.json | nuctl invoke hf-detr-v1-coco -c 'application/json'
