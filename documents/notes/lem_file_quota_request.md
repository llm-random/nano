# LEM file quota request

Email to the LEM admins (WCSS) asking for a permanent raise of project 649's file quota to 2M files or more.
Drafted 2026-11-20, not sent yet.

```text
Subject: File quota increase for grant plgllmefficont3 (/lustre/pd03, project 649)

Dear LEM Support Team,

I am writing on behalf of our team working under the plgllmefficont3 grant. Our project directory, /lustre/pd03/plgrid/plgllmefficont3 (Lustre project 649), has a generous disk quota of 29.3 TB, of which we use about 16 TB. The file limit, however, is 1,048,576 files, and we reach it long before we come close to the space limit.

Several of us submit hundreds of experiments each, and every run leaves its own code snapshot, logs and checkpoints behind. On 19 November we hit the limit, and jobs that had been training for hours failed with "Disk quota exceeded" while saving their final checkpoints. We have cleaned up since and are back to about 763,000 files, but that headroom will not last long.

We are now planning a large series of short experiments, each of which writes many log files. We also noticed that the MDT of /lustre/pd03 is nearly full (99% in lfs df -i). We understand that file counts are a shared constraint, so we are asking for a moderate increase.

We would like to ask for a permanent increase of our file quota to 2 million files, or more if possible. On our side, we will also reduce our file count, for example by packing old results that we keep only as a historical record into archives.

Thank you very much for your help.

Best regards,
Maciej Stefaniak
PLGrid login: plgmstefaniak
```

## The numbers behind it (LEM, 2026-11-20)

| What | Value | Source |
|---|---|---|
| Project 649 files | 763,120 of 1,048,576 (soft limit 900,096, 1-week grace) | `lfs quota -h -p 649 /lustre/pd03` |
| Project 649 space | 15.98T of 29.3T (soft limit 24.56T) | same |
| pd03 metadata target (one MDT, ZFS) | 74.3M of 75.6M inodes, 7.4 of 7.5T used, 131G free (99%) | `lfs df -i /lustre/pd03`, `lfs df -h` |
| Groups sharing pd03 | 147 directories under /lustre/pd03/plgrid | `ls /lustre/pd03/plgrid` |
| 11-19 job failures | `OSError: [Errno 122] Disk quota exceeded: 'checkpoint'` (project quota, not a full MDT) | `slurm-6353710_0.out` |

A quota of 2M lets the team add about 1.24M files. `lfs df -i` estimates about 1.28M free inodes in
the whole filesystem, so the promise to reduce our file count may decide the answer.
