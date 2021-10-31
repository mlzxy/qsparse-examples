set -e
trap "exit" INT
export PYTHONPATH=../../pypi:$PYTHONPATH
for mode in "$@"; do
	cmd="python3 train.py --train-file data/91-image_x3.h5  --eval-file data/Set5_x3.h5   --scale 3  --lr 1e-3 --batch-size 16  --num-epochs 200  --num-workers 8  --seed 123 "
	suffix=qsparse/super_resolution/$mode/
	if [[ -z "${COLAB_GPU}" ]]; then
		echo "$cmd"
		flagfile=$HOME/Desktop/$suffix/finish.txt
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, SKIP. \n\n\n"
		else
			mkdir -p  $HOME/Desktop/$suffix 
			$cmd --train-mode $mode --outputs-dir  $HOME/Desktop/$suffix 
			echo "finish" >$flagfile
		fi
	fi
done

