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
alpha=1
beta=0.05
anchor_count=6


fedvfa_alpha=0.01
classifier_iter=1

#python main.py $param --loss CE
#python main.py $param --loss FedRs
#python main.py $param --loss FedLC
#python main.py $param --loss FedProx
#python main.py $param --loss MoonLoss
#python main.py $param --loss FedFA
python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter


dirichlet_alpha=0.5
param="--dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"


#python main.py $param --loss CE
#python main.py $param --loss FedRs
#python main.py $param --loss FedLC
#python main.py $param --loss FedProx
#python main.py $param --loss MoonLoss
#python main.py $param --loss FedFA
#python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter


dirichlet_alpha=0.2
param="--dataset $dataset --model $model --epochs $epochs --learning_rate $learning_rate --dirichlet_alpha $dirichlet_alpha --batch_size $batch_size --weight_decay $weight_decay --local_epochs $local_epochs --print_acc_interval $print_acc_interval"

#python main.py $param --loss CE
#python main.py $param --loss FedRs
#python main.py $param --loss FedLC
#python main.py $param --loss FedProx
#python main.py $param --loss MoonLoss
#python main.py $param --loss FedFA
#python main.py $param --loss FedVFA --alpha $fedvfa_alpha --classifier_iter $classifier_iter
