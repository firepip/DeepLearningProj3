 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -J MoreRandom
 #BSUB -gpu "num=1"
 #BSUB -n 1
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
 #Arg4 = Model: 1 - impala, 2 - leaky impala, 3 - 5 blocks, 4 - 2 blocks
 #Arg5 = Data augmentation strategy, 0=identity, 1=crop, 2=translate, 3=cutout, 4=colormix, 5=random sequences of all
 #Arg6 = Number of features (output of IMPALA)
 #Arg7 = Validation augmentation strategy. Can be 0-4 or not specified. When specified, this strategy will be removed from the random rounds of Arg5.
 python3 train.py 10 1 coinrun 1 0 512 0
 echo "Done..."