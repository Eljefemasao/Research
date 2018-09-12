# download.sh
if [ $# -ne 2 ]; then
  exit 1
fi

wget $2 -T 5 -t 5 -nc -b -a wget.log
