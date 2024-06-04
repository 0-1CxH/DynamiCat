# get pardir of pardir of pardir this file as the root of the project
project_root=$(dirname $(dirname $(dirname $(readlink -f "$0"))))
echo "Project root: $project_root"
# get the tokenization python file
target_python_file="$project_root/dynamicat/tokenization/main_tokenize.py"
echo "Target python file: $target_python_file"

# use getopts to parse the arguments
while getopts ":i:t:" args; do
  case $args in
    i)
      metadata_file_path=$OPTARG
      ;;
    t)
      tokenizer_path=$OPTARG
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done

# print args
echo "Metadata file path: $metadata_file_path"
echo "Tokenizer path: $tokenizer_path"
# print config file in the tokenizer path
cat $tokenizer_path/config.json

# Run the tokenization pipeline
python $target_python_file \
  --dataset_metadata_path $metadata_file_path \
  --tokenizer_path $tokenizer_path