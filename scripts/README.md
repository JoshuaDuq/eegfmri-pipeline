# scripts/

This folder contains helper scripts to run the pipeline on a remote GCP VM while keeping inputs and visualization local.

## One-time setup

### 1) Install + auth `gcloud` (local)

```bash
brew install --cask google-cloud-sdk
export PATH=/opt/homebrew/share/google-cloud-sdk/bin:"$PATH"

gcloud auth login
gcloud config set project <PROJECT_ID>
```

### 2) Identify the VM (local)

```bash
gcloud compute instances list
```

Record:

- **Instance name**
- **Zone**
- **External IP**

### 3) Set up `ssh` alias (local)

Add to `~/.ssh/config`:

```sshconfig
Host thermal-gcp
  HostName <VM_EXTERNAL_IP>
  User joduq24
  IdentityFile ~/.ssh/google_compute_engine
  IdentitiesOnly yes
```

Verify:

```bash
ssh thermal-gcp "hostname && whoami"
```

### 4) Mount the data disk to `/mnt/data` (remote)

If you attached a persistent disk at `/dev/sdb`:

```bash
ssh thermal-gcp
sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data
sudo chown -R joduq24:joduq24 /mnt/data
sudo blkid /dev/sdb
sudo bash -lc 'echo "UUID=<UUID_FROM_BLKID> /mnt/data ext4 defaults,nofail 0 2" >> /etc/fstab'
sudo umount /mnt/data
sudo mount -a
df -h /mnt/data
exit
```

### 5) Create remote repo dir + install deps (remote)

```bash
ssh thermal-gcp "sudo mkdir -p /mnt/data/Thermal_Pain_EEG_Pipeline && sudo chown -R joduq24:joduq24 /mnt/data"

ssh thermal-gcp "sudo apt-get update && sudo apt-get install -y rsync python3-venv python3-pip"
```

### 6) First sync + create the VM venv (local → remote)

```bash
export REMOTE_HOST=thermal-gcp
./scripts/gcp.sh sync
```

Create a Linux venv on the VM (do not use a synced macOS venv):

```bash
ssh thermal-gcp 'cd /mnt/data/Thermal_Pain_EEG_Pipeline && rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt'
```

## Every run

### 1) Start VM (local)

```bash
gcloud compute instances start eegpipeline --zone us-central1-a --project eegpipeline-481605
```

### 2) Update SSH alias IP if needed (local)

```bash
gcloud compute instances list --project eegpipeline-481605
```

### 3) Verify SSH (local)

```bash
ssh thermal-gcp "hostname && whoami"
```

### 4) Sync + run (local)

```bash
export REMOTE_HOST=thermal-gcp
export REMOTE_VENV_ACTIVATE=/mnt/data/Thermal_Pain_EEG_Pipeline/.venv/bin/activate
export REMOTE_N_JOBS=-1

./scripts/gcp.sh sync
./scripts/gcp.sh batch features compute --subject 0001
```

### 5) Stop VM (local)

```bash
gcloud compute instances stop eegpipeline --zone us-central1-a --project eegpipeline-481605
```

### 6) Run overnight (auto-stop)

```bash
export REMOTE_HOST=thermal-gcp
export REMOTE_VENV_ACTIVATE=/mnt/data/Thermal_Pain_EEG_Pipeline/.venv/bin/activate
export REMOTE_N_JOBS=-1

export GCP_PROJECT=eegpipeline-481605
export GCP_ZONE=us-central1-a
export GCP_INSTANCE=eegpipeline

./scripts/gcp.sh batch-stop features compute --subject 0001
```
