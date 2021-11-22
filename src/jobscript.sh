 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -J Procgen
 #BSUB -gpu "num=1"
 #BSUB -n 6
 #BSUB -W 24:00 
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o Output_%J.out
 #BSUB -e Error_%J.err
 #BSUB -u patrickvibild@gmail.com
 #BSUB -B 
 #BSUB -N 
 echo "Running script..."
 #Arg1 = Total steps
 #Arg2 = Number of levels
 #Arg3 = Game, default = starpilot
 #Arg4 = Model
 #Arg5 = Data augmentation strategy, 0=identity, 1=crop, 2=translate, 3=cutout, 4=colormix, 5=random sequences of all
 python3 train.py 10 10000 starpilot 1 0
 echo "Done..."