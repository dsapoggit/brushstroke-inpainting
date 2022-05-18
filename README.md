# brushstroke-inpainting
HSE AMI 4th year thesis


Настройка окружения 

0. Необходимо клонировать репрозиторий с LaMa для создания инпеинтингов 

git clone https://github.com/saic-mdal/lama.git

и проследовать инструкции по настройке окружения с https://github.com/saic-mdal/lama


1. Pytorch >= 1.8
2. conda install pytorch-cluster -c pyg
3. pip install git+https://github.com/justanhduc/neural-monitor - для логгирования

Fill the mask - python .\create_mask.py path_to_image x_coord_of_mask y_coord_of_mask
Inpainting - python3 lama/bin/predict.py model.path=$(pwd)/lama/big-lama indir=$(pwd)/input outdir=$(pwd)/output
Brushstroke - python .\brush_stroke_base.py path_to_content path_to_style
Style Transfer -  python .\style_transfer.py path_to_content path_to_style
