# GENERALIZATION IN VIDEO GAMES: IMPROVED GENERALIZATION USING DATA AUGMENTATION

Repository contains code for benchmarking different data augmentation techniques in Procgen benchmark environment.

### Installation and Requirements.

```
git clone https://github.com/firepip/DeepLearningProj3/commits/main
pip install procgen
```

The code requires GPU-accelerated machine. 

### Execute code in HPC.

1. Log into HPC using a ssh shell. 
`ssh <student ID>@login2.gbar.dtu.dk`
2. Load next modules inside HPC
```
Loaded module: python3/3.8.9
Loading python3/3.8.9
Loaded module: cuda/8.0
Loaded module: cudnn/v7.0-prod-cuda8
Loaded module: ffmpeg/4.2.2
```
3. Install Procgen `pip install procgen`
4. Copy the source code into HPC (see next section)
5. Change any required setup in the shell file src/jobscript.sh
6. Start the job with the command 
```
cd src
bsub < jobscript.sh
```

### How to copy files into HPC and obtain the generated log files and videos
#### Linux
Will mount HPC directory into the `local-directory`
```
sshfs username@transfer.gbar.dtu.dk: local-directory
```
#### Command line transfer
Copies the current directory into 
Linux SCP
```
scp -r . <student ID>@login2.gbar.dtu.dk:/remote/folder/
```
Windows PSCP (requires installation)
```
pscp scp -r . <student ID>@login2.gbar.dtu.dk:/remote/folder/
```
