# Scenario 1:
2 projections separees de 90 degres.
La minimisation ne peut aboutir car il existe une droite du plan des parametres ou la fonction de cout est minimale.
Ceci est du a la raison suivante: appelons baseline la droite joignant la position de source 1 et la position de source 2' (i.e. la position de source 2 translatee du montant de la translation de l'objet). Si l'on deplace le gantry 2 le long de cette baseline, la trace des plans epiloires sur le detecteur 2 est invariante, donc les profils de consistance (les G_f(/theta)) restent inchanges et toute position de source 2 est 'aussi coherente avec la premiere projection' que la "vraie" position 2'.
Le scenario a 2 projections ne peut donc pas deboucher.

# Scenario 2:
3 projections. A ce stade, il faudrait entrer un peu plus dans la geometrie d'acquisition... 
Un premier essai a 0, 90, 180 degres (avec deplacement ech entre 0 et 90 deg) revele un aspect inattendu:
La paire (0,180) semble coherente du point de vue Grangeat (un offset entre les 2 profils du point de vue FB del'ordre de 30.4 +/- 0.07).
Or, il y a une difference de grossissement entre les 2 projections... 
ATTENTION : dans le cas (0,180), les epipoles sont dans les images... pb de ponderation (au moins en FB ???)

Voir aussi le papier de Tobias Wurfl (ICCV 2019) sur le calcul de la matrice fondamentales a partir des DCC. La matrice fondamentale encode la position/orientation respective de 2 gantry (identiques, i.e. parametres internes identiques).

Pour le scenario 2, je prends 2 trajectoire qui sont completes en short scan et compte tenu du rapprochement de l'ech de la source.
Avec un phantom scale de 40, l diam de l'ech est <50. Avec une position 1 a (0,200), un sid de 500, je prends l'arc 1 z>150 (geom RTK), i.e. -alpha, alpha avec alpha = arccos(150/500).

Mardi 18/06:
Mise au point du code avec 3 projets/2paires. Ca fonctionne. Mais 
- une paire "proche" de 180 degres s'eloigne un peu de la bonne solution
- il y a toujours une "vallee" dans la fonction de cout 2d...
- Tentatives de minimisation : echec. Nelder-Mead echoue. Powell fait a peu pres le job.

To be continued.

Papier : High-resolution CT reconstruction from Zoom-In Partial Scans (ZIPS) with simultaneous estimation of inter-scan ROI motion, Heneda et al., SPIE2022.

Scenario de reference :
Avec (0,72) et (0,108) en deg et la meme deplacement de (0,200), la fonction de cout admet un min unique.
Powell le trouve assez bien (-0.598, 200.1) avec fb quelque soit l'init. et (-0.427,200.0) avec gr.


Questions en suspens :
- Si les paires s'eloignent de la aposition 90 degre, degradation ?
- Si on prend deux paires 0 90 (0,90 et 0,270), est-ce que c'est mieux ?
- Role du vecteur de deplacement ?


La paire (0,171) pose pb car l'epipole est dans les 2 detecteurs.
La paire (0,28) pose aussi probleme, mais je ne sais pas pourquoi.... Le signal sur p0 est nul (c'est la projection fanbeam de l'im0 ponderee qui renvoie 0.) Plus chiant a tracker...


Scenario Final:
2 arcs 0-9- et 90-180, petit objet (5cm de rayon), large detecteur (1m approx!). On approche l'obj de 20 cm (/sid = 57). Taille obj et largeur det necessaire pour que ca reste dans le fov.

Pb de selection des paires:
- 45-135 => paire consistente (ne devrait pas pouvoir etre plus bas).

Etude bruit vs nombre de pairs
-------------------------------
Pas si isimple. 
- Premier probleme : 1 paire qui parait admissible (i.e. dans la zone de la carte 2d de la consistence initiale) n'est pas forcement correcte lorsqu'on evalue la consistence pour un deplacement donne. Notamment, si le deplacement (0,284) est applique e.g. et si la paire est disons 80,120, la baseline peut beaucoup changer et ne plus etre dans une configuration "standard".
- Second probleme: meme en supprimant ces pairs (j-i<40) la fonction de cout ne se comporte pas bien avec * paires ou plus, quelque soit le niveau de bruit. Et je ne comprnds pas pourquoi.



