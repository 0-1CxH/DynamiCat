# get pardir of pardir of pardir this file as the root of the project
project_root=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "Project root: $project_root"
# get the tokenization python file
target_python_file="$project_root/dynamicat/tensorplanning/main_tensor_plan.py"
echo "Target python file: $target_python_file"

# use getopts to parse the arguments
while getopts ":i:c:r:" args; do
  case $args in
    i)
      input_path=$OPTARG
      ;;
    c)
      tensor_parameter_count_limit=$OPTARG
      ;;
    r)
      plan_order_type=$OPTARG
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done

if [ -z "$plan_order_type" ]; then
  # Run the planning pipeline
  # output_path is input_path(no suffix)+"_GPUMemoryRestricted_"+tensor_parameter_count_limit+"_Order_"+plan_order_type+".pt"
  output_path=${input_path%.*}"_GPUMemoryRestricted_"$tensor_parameter_count_limit"_Order_none.pt"
  # print args
  echo "Input path: $input_path"
  echo "Output path: $output_path"
  echo "Tensor parameter count limit: $tensor_parameter_count_limit"
  echo "Plan order type: $plan_order_type"
  python $target_python_file \
    --tensor_planner_type "GPUMemoryRestricted" \
    --input_path $input_path \
    --output_path $output_path \
    --tensor_parameter_count_limit $tensor_parameter_count_limit
else
  # Run the planning pipeline
  # output_path is input_path(no suffix)+"_GPUMemoryRestricted_"+tensor_parameter_count_limit+"_Order_"+plan_order_type+".pt"
  output_path=${input_path%.*}"_GPUMemoryRestricted_"$tensor_parameter_count_limit"_Order_"$plan_order_type".pt"
  echo "Input path: $input_path"
  echo "Output path: $output_path"
  echo "Tensor parameter count limit: $tensor_parameter_count_limit"
  python $target_python_file \
    --tensor_planner_type "GPUMemoryRestricted" \
    --input_path $input_path \
    --output_path $output_path \
    --tensor_parameter_count_limit $tensor_parameter_count_limit \
    --plan_order_type $plan_order_type
fi
