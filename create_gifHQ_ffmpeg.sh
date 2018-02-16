ffmpeg -start_number 230 -i outputs/temp/mpi_gif_vmem%03d.png -vf palettegen -y palette.png
ffmpeg -start_number 230 -framerate 20 -loop 0 -i outputs/temp/mpi_gif_vmem%03d.png -i palette.png -lavfi paletteuse -y hq_gif_vmem.gif
rm outputs/temp/gif_vmem*.png
