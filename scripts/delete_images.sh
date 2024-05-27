#!/bin/bash

# Check if a directory was provided as input
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

# Assign the provided directory to a variable
DIRECTORY=$1

# Verify that the directory exists
if [ ! -d "$DIRECTORY" ]; then
  echo "Error: Directory does not exist."
  exit 1
fi

# List all image files to be deleted for confirmation
echo "The following image files will be deleted:"
find "$DIRECTORY" -type f \( -iname \*.jpg -o -iname \*.png -o -iname \*.gif -o -iname \*.jpeg -o -iname \*.bmp -o -iname \*.tif -o -iname \*.tiff \)

# Ask for user confirmation
read -p "Are you sure you want to delete all these files? (y/n) " -n 1 -r
echo    # Move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]
then
  # Delete the files
  find "$DIRECTORY" -type f \( -iname \*.jpg -o -iname \*.png -o -iname \*.gif -o -iname \*.jpeg -o -iname \*.bmp -o -iname \*.tif -o -iname \*.tiff \) -delete
  echo "Files deleted successfully."
else
  echo "File deletion cancelled."
fi
