#!/bin/bash

files=(
~/Meshes/MANIX_cortado.ply
~/Meshes/0543_pulmao.ply
~/Meshes/TOUTATIX_cortado.ply
~/Meshes/cerebro_metade.ply
~/Meshes/MANIX_cortado_50000.ply)

metodos=(MIP AIP MIDA RAYCASTING MINIP)

for i in $(seq 3); do
    for m in ${metodos[@]}; do
        for ply_file in ${files[@]};  do
            echo "\n========================================================================\n"
            echo $m
            echo $ply_file
            time python triangle_texture_2.py -i "$ply_file" -o /tmp/saida.ply -t ~/voldatasets/manix_512x512x460_049x049x07.raw -s 0.48828125 0.48828125 0.7000120000000152 -d 512 512 460 -c ~/Sources/github/invesalius3/presets/raycasting/color_list/HotMetal.plist --ww 750 --wl 750 --offset 3 --nslices 10 -p 5000 -m $m
            echo "\n========================================================================\n"
        done
    done
done

# echo "Manix"
# time python triangle_texture_2.py -i ~/Meshes/MANIX_cortado.ply -o /tmp/saida.ply -t ~/voldatasets/manix_512x512x460_049x049x07.raw -s 0.48828125 0.48828125 0.7000120000000152 -d 512 512 460 -c ~/Sources/github/invesalius3/presets/raycasting/color_list/HotMetal.plist --ww 750 --wl 750 --offset 3 --nslices 10 -p 5000

# echo "\n========================================================================\n"
# echo "Pulmao"

# time python triangle_texture_2.py -i ~/Meshes/0543_pulmao.ply -o /tmp/saida.ply -t ~/voldatasets/0543_cortado_512x512x300_0.5859375_0.5859375_1.0.i16.dat -s 0.5859375 0.5859375 1.0 -d 512 512 300 -c ~/Sources/github/invesalius3/presets/raycasting/color_list/HotMetal.plist --ww 750 --wl 750 --nslices 10 -p 5000

# echo "\n========================================================================\n"
# echo "TOUTATIX"

# time python triangle_texture_2.py -i ~/Meshes/TOUTATIX_cortado.ply -o /tmp/saida.ply -t ~/voldatasets/coracao_309x512x512_0.33203125_0.33203125_0.5.i16.dat -s 0.33203125 0.33203125 0.5 -d 512 512 309 -c ~/Sources/github/invesalius3/presets/raycasting/color_list/HotMetal.plist --ww 750 --wl 750 --offset 3 --nslices 10 -p 5000

# echo "\n========================================================================\n"
# echo "Cerebro"

# time python triangle_texture_2.py -i ~/Meshes/cerebro_metade.ply -o /tmp/saida.ply -t ~/voldatasets/cerebro_256x256x180_0.9992523193359943_1.0_1.0.i16.dat -s 0.9992523193359943 1.0 1.0 -d 180 256 256 -c ~/Sources/github/invesalius3/presets/raycasting/color_list/HotMetal.plist --ww 750 --wl 750 --offset 3 --nslices 10 -p 5000

# echo "\n========================================================================\n"
# echo "Manix reduced"

# time python triangle_texture_2.py -i ~/Meshes/MANIX_cortado_50000.ply -o /tmp/saida.ply -t ~/voldatasets/manix_512x512x460_049x049x07.raw -s 0.48828125 0.48828125 0.7000120000000152 -d 512 512 460 -c ~/Sources/github/invesalius3/presets/raycasting/color_list/HotMetal.plist --ww 750 --wl 750 --offset 3 --nslices 10 -p 5000
