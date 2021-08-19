if [ "$#" -ne 2 ];
then
  echo "download_zenodo.sh <data list file> <download folder>";
  exit -1;
fi
input="$1"
output="$2"
mkdir -p "$output"
while IFS=$'\t' read -r id url
do
  echo $id $url
  newdir="$output/$id"
  mkdir -p $newdir
  pushd $newdir
  WGETCMD="wget -t0 -c '"$url"'"
  eval "$WGETCMD"
  popd
done < "$input"
echo "----- Script Complete -----"
