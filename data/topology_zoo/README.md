To generate images of these networks, run something like

```bash
for d in ../data/topology_zoo/*; do if [ -d "$d" ]; then echo "$d"; py plot_network.py -p "$d"/probes.csv -l "$d"/connectivity.csv -o networks/"${d##*/}".png; fi; done
```

from `src/`