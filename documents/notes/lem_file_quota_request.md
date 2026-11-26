# LEM file quota request

Reply to the LEM admins (WCSS), who announced lowering the file quota to 200,000: it asks for a permanent
raise of project 649's quota to 2M files instead. Drafted 2026-11-20, revised 2026-11-26, not sent yet.

```text
Subject: File quota increase for grant plgllmefficont3 (/lustre/pd03, project 649)

Dear LEM Support Team,

We received your message that the file quota will be lowered to 200,000 files. I am writing on behalf of our group in the plgllmefficont3 grant, because this limit would be far below what our work needs. We are a large research team working on LLM pretraining, with multiple projects running concurrently.

Our project directory, /lustre/pd03/plgrid/plgllmefficont3 (Lustre project 649), has a generous disk quota of 29.3 TB, of which we use about 17 TB. The file limit, however, is 1,048,576 files, and we reach it long before we come close to the space limit.

Several of us submit hundreds of experiments each, and every run leaves its own code snapshot, logs and checkpoints behind. On 19 November we hit the limit, and jobs that had been training for hours failed with "Disk quota exceeded" while saving their final checkpoints. We cleaned up afterwards, but within a week we were back above 900,000 files.

We noticed that the MDT of /lustre/pd03 is nearly full (97% in lfs df -i). We understand that file counts are a shared constraint, so we are asking for a moderate increase.

Instead of the reduction, we would like to ask for a permanent increase of our file quota to 2 million files. On our side, we will also reduce our file count, for example by packing old results that we keep only as a historical record into archives.

Thank you very much for your help.

Best regards,
Maciej Stefaniak
PLGrid login: plgmstefaniak
```

## The numbers behind it (LEM, 2026-11-26)

| What | Value | Source |
|---|---|---|
| Admins' announcement | file quota to be lowered to 200,000 | their email to the team |
| Project 649 files | 913,325 of 1,048,576: over the soft limit (900,096), writes stop when the 1-week grace ends (6d9h left on 11-26) | `lfs quota -h -p 649 /lustre/pd03` |
| Project 649 space | 17.1T of 29.3T (soft limit 24.56T) | same |
| pd03 metadata target (one MDT, ZFS) | 73.0M of 75.5M inodes, 7.2 of 7.4T used, 257G free (97%; 99% on 11-20) | `lfs df -i /lustre/pd03`, `lfs df -h` |
| Groups sharing pd03 | 147 directories under /lustre/pd03/plgrid (11-20) | `ls /lustre/pd03/plgrid` |
| 11-19 job failures | `OSError: [Errno 122] Disk quota exceeded: 'checkpoint'` (project quota, not a full MDT) | `slurm-6353710_0.out` |

A quota of 2M lets the team add about 1.09M files. `lfs df -i` estimates about 2.55M free inodes in
the whole filesystem, so the promise to reduce our file count may decide the answer.
