# get pardir of pardir of pardir this file as the root of the project
project_root=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "Project root: $project_root"
# get the tokenization python file
target_python_file="$project_root/dynamicat/tensorplanning/main_tensor_plan.py"
echo "Target python file: $target_python_file"

# use getopts to parse the arguments
while getopts ":i:k:d:b:" args; do
  case $args in
    i)
      input_path=$OPTARG
      ;;
    k)
      primary_key=$OPTARG
      ;;
    d)
      plan_length_diff=$OPTARG
      ;;
    b)
      max_plan_size=$OPTARG
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done



# output_path is input_path(no suffix)+"_PlanLengthDiff_"+plan_length_diff+"_MaxPlanSize_"+max_plan_size+".pt"
output_path=${input_path%.*}"_PlanLengthDiff_"$plan_length_diff"_MaxPlanSize_"$max_plan_size".pt"

# print args
echo "Input path: $input_path"
echo "Output path: $output_path"
echo "Primary key: $primary_key"
echo "Plan length diff: $plan_length_diff"
echo "Max plan size: $max_plan_size"

# Run the planning pipeline
python $target_python_file \
  --tensor_planner_type "LengthDifferenceRestricted" \
  --input_path $input_path \
  --output_path $output_path \
  --primary_key $primary_key \
  --max_token_diff $plan_length_diff \
  --batch_size $max_plan_size
