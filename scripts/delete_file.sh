#!/bin/bash

declare -a keep_files
directory=""

while getopts "d:f:" opt; do
  case $opt in
    d) directory=$OPTARG ;;
    f) 
      if [[ "$OPTARG" == "*" ]]; then
        # If the argument is '*', set keep_files to empty
        keep_files=()
      else
        keep_files+=("$OPTARG")
      fi
      ;;
    *) exit 1 ;;
  esac
done

if [[ -z "$directory" ]]; then
  echo "Error: Directory is not specified."
  exit 1
fi

cd "$directory" || exit

if [[ ${#keep_files[@]} -eq 0 ]]; then
  echo "Deleting all files in directory: $directory"
  rm -f *  # Remove all files in the directory
else
  for file in *; do
    if [[ ! " ${keep_files[*]} " =~ " ${file} " ]]; then
      rm "$file"
    fi
  done
fi
