# NeuroevolutionAlgorithm_ML
Creating a Neuroevolutionary Algorithm in python

Created By: <br/>
---- Name: David De Angelis<br/>
---- Student ID: 12913873<br/>
---- GitHub: david-de-angelis<br/>
---- Date: 02/10/2021<br/>

Created For:<br/>
---- University: University of Technology Sydney <br/>
---- Faculty: Faculty of Engineering and Information Technology (FEIT) <br/>
---- Subject: (31005) Machine Learning <br/>
---- Assessment: Assessment task 2: Algorithm Implementation and Report <br/>

Acknowledgements: <br/>
---- Daniel Shiffman (The Coding Train - YT) for his explanation of how Neuro-evolution works, <br/>

Disclaimer: <br/>
---- All code was written by me, with the exception of the following functions: <br/>
---- ---- main.py shuffle() <br/>
---- ---- ---- https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison <br/>
---- ---- GeneticAlgorithm.py pickOne()  <br/>
---- ---- ---- https://github.com/CodingTrain/website/blob/main/CodingChallenges/CC_035.4_TSP_GA/P5/ga.js line 54. <br/>
---- ---- ---- adapted from javascript to python, and to work with my given setup <br/>

Pre-requisites: <br/>
---- Must have python3 <br/>
---- ---- confirm with ('python3 --version') <br/>
---- ---- I used Python 3.9.7 for development <br/>
---- Must have the following packages installed:< br/>
---- ---- numpy ('pip install numpy') <br/>
---- ---- sklearn ('pip install -U scikit-learn') <br/>

How to run: <br/>
---- 'python3 main.py' --runs with default "iris" dataset <br/>
---- 'python3 main.py {dataset}' <br/>
---- ---- possible datasets: "iris" (easy), "digits" (hard) <br/>
---- ----  e.g. 'python3 main.py digits' <br/>

Notes: <br/>
---- Highest achieved accuracy: <br/>
---- ---- Iris: 98% (after ~50 generations) <br/>
---- ---- Digits: 85% (after ~800 generations) <br/>
