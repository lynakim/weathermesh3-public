INSTANCE=aws

rsync -avz --progress /fast/consts $INSTANCE:/fast/
rsync -avz --progress /fast/optimizers $INSTANCE:/fast/