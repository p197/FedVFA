cd ..

current_path=$(pwd)
echo "当前工作路径为：$current_path"


batch_size=64
learning_rate=0.01
epochs=1500
model="ResNet20"
dataset="svhn"

weight_decay=0
local_epochs=1
print_acc_interval=10

dirichlet_alpha=0.1
param="--dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

classifier_iter=1

beta=0.1

alpha=0.001
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --log_file_name $log_file_name

alpha=0.005
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --log_file_name $log_file_name

alpha=0.01
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --log_file_name $log_file_name

alpha=0.05
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --log_file_name $log_file_name

alpha=0.1
log_file_name="beta=$beta,alpha=$alpha"
python main.py $param --loss FedVFA --alpha $alpha --beta $beta --classifier_iter $classifier_iter --log_file_name $log_file_name

