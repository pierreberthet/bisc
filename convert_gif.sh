convert -resize 90% -delay 13 -loop 0 outputs/temp/mpi*vmem*.png outputs/mpi_vmem.gif
convert -resize 90% -delay 13 -loop 0 outputs/temp/mpi*vext*.png outputs/mpi_vext.gif
rm outputs/temp/*.png
