# LEM file quota request

Reply to the LEM admins (WCSS), who announced lowering the file quota to 200,000. It asks for a permanent
raise of project 649's quota to 2M files, with keeping 1M as the fallback. In Polish, with the team's
additions of 11-26: agreed with the grant leader, team size, GPU-hours, the 1M fallback. Drafted 2026-11-20,
revised 2026-11-26, not sent yet. The last English version is in commit bbc34df5.

```text
Temat: Limit liczby plików dla grantu plgllmefficont3 (/lustre/pd03, projekt 649)

Dzień dobry,

otrzymaliśmy Państwa wiadomość o planowanym obniżeniu limitu liczby plików do 200 000. Piszę w imieniu naszego zespołu z grantu plgllmefficont3, ponieważ taki limit byłby dla nas zdecydowanie za niski. Prośba jest uzgodniona z kierownikiem grantu.

Jesteśmy dużym zespołem badawczym, który zajmuje się pretrenowaniem dużych modeli językowych (LLM) i prowadzi równolegle wiele projektów. Jest nas wielu – do grantu należy 14 osób.

Mamy do dyspozycji 100 000 godzin GPU, a nasz katalog projektu, /lustre/pd03/plgrid/plgllmefficont3 (projekt Lustre 649), ma duży limit przestrzeni dyskowej (29,3 TB), z czego wykorzystujemy ok. 17 TB. Limit liczby plików wynosi jednak 1 048 576 i dochodzimy do niego na długo przed wyczerpaniem miejsca.

Uruchamiamy setki eksperymentów, a każdy z nich zostawia własne logi i checkpointy. 19 listopada osiągnęliśmy limit. Usunęliśmy wtedy część plików, ale w ciągu tygodnia znów przekroczyliśmy 900 000.

Zauważyliśmy, że MDT systemu plików /lustre/pd03 jest prawie pełny (97% według lfs df -i). Rozumiemy, że liczba plików to zasób współdzielony, dlatego prosimy o umiarkowane zwiększenie limitu.

Prosimy o trwałe zwiększenie limitu liczby plików do 2 milionów. Jeśli nie będzie to możliwe, prosimy przynajmniej o pozostawienie obecnego limitu 1 miliona plików. Ze swojej strony ograniczymy też liczbę plików, na przykład pakując do archiwów starsze wyniki, które przechowujemy jedynie jako zapis historyczny.

Bardzo dziękujemy za pomoc.

Z poważaniem
Maciej Stefaniak
login PLGrid: plgmstefaniak
```

## The numbers behind it (LEM, 2026-11-26)

| What | Value | Source |
|---|---|---|
| Admins' announcement | file quota to be lowered to 200,000 | their email to the team |
| Grant members | 14 users on the Slurm account plgllmefficont3, 6 of them ran jobs in November | `sacctmgr show assoc account=plgllmefficont3`, `sacct -a -A plgllmefficont3` |
| GPU-hours | 30,657 of 100,000 used (30.7%), about 69,000 left | `service-balance` |
| Project 649 files | 913,325 of 1,048,576: over the soft limit (900,096), writes stop when the 1-week grace ends (6d9h left on 11-26) | `lfs quota -h -p 649 /lustre/pd03` |
| Project 649 space | 17.1T of 29.3T (soft limit 24.56T) | same |
| pd03 metadata target (one MDT, ZFS) | 73.0M of 75.5M inodes, 7.2 of 7.4T used, 257G free (97%; 99% on 11-20) | `lfs df -i /lustre/pd03`, `lfs df -h` |
| Groups sharing pd03 | 147 directories under /lustre/pd03/plgrid (11-20) | `ls /lustre/pd03/plgrid` |
| 11-19 job failures | `OSError: [Errno 122] Disk quota exceeded: 'checkpoint'` (project quota, not a full MDT) | `slurm-6353710_0.out` |

A quota of 2M lets the team add about 1.09M files. `lfs df -i` estimates about 2.55M free inodes in
the whole filesystem, so the promise to reduce our file count may decide the answer.
