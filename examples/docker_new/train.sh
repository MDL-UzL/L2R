
task_name=$1
num_iterations=$2
echo "Task ${task_name}, Iterations ${num_iterations}"
keypts_dir="../Learn2Reg_Dataset_release_v1.1/${task_name}/keypointsTr"
echo ${keypts_dir}
if [ -d "$keypts_dir" ]; then
    echo "Training paired network with keypoint loss"
    python train_registration_paired.py ${task_name} ${num_iterations} 0
fi
    echo "Training label-based networks"
    python train_segment.py ${task_name} ${num_iterations} 0
    python train_registration.py ${task_name} ${num_iterations} 0