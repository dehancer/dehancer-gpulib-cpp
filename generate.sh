#!/bin/bash

source_dir="$1"
destination_dir="$2"

[[ -d "$destination_dir" ]] || mkdir -p "$destination_dir"
cd "$destination_dir"

for file in $(find $source_dir -name "*.png"); do
  image_name="dehancer_$(basename -s .png $file)"

  cp $file "$destination_dir/$image_name"

  xxd -i "$image_name" "${image_name}.c"

  rm -f "$image_name"
done
