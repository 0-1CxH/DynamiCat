# get pardir of pardir of pardir this file as the root of the project
project_root=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "Project root: $project_root"
# get the tokenization python file
target_python_file="$project_root/dynamicat/tensorplanning/main_tensor_plan.py"
echo "Target python file: $target_python_file"

# use getopts to parse the arguments
while getopts ":i:b:s:" args; do
  case $args in
    i)
      input_path=$OPTARG
      ;;
    b)
      batch_size=$OPTARG
      ;;
    s)
      enable_smart_batching=$OPTARG
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done

# output_path is input_path(no suffix)+"_FixedBatchSize_"+batch_size+".pt"
output_path=${input_path%.*}"_FixedBatchSize_"$batch_size".pt"


# print args
echo "Input path: $input_path"
echo "Output path: $output_path"
echo "Batch size: $batch_size"
echo "Enable smart batching: $enable_smart_batching"

# if enable_smart_batching is not provided, do not set it
# else add to the command
if [ -z "$enable_smart_batching" ]; then
  # Run the planning pipeline
  python $target_python_file \
    --tensor_planner_type "FixedBatchSize" \
    --input_path $input_path \
    --output_path $output_path \
    --batch_size $batch_size
else
  # Run the planning pipeline
  python $target_python_file \
    --tensor_planner_type "FixedBatchSize" \
    --input_path $input_path \
    --output_path $output_path \
    --batch_size $batch_size \
    --enable_smart_batching
fi



