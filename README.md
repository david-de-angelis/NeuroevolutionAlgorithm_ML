# NeuroevolutionAlgorithm_ML
Creating a Neuroevolutionary Algorithm in python

Created By: 
    - Name: David De Angelis
    - Student ID: 12913873
    - GitHub: david-de-angelis
    - Date: 02/10/2021

Created For:
    - University: University of Technology Sydney
    - Faculty: Faculty of Engineering and Information Technology (FEIT)
    - Subject: (31005) Machine Learning
    - Assessment: Assessment task 2: Algorithm Implementation and Report

Acknowledgements:
    - Daniel Shiffman (The Coding Train - YT) for his explanation of how Neuro-evolution works, 

Disclaimer:
    - All code was written by me, with the exception of the following functions:
        - main.py shuffle() 
            - https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
        - GeneticAlgorithm.py pickOne()  
            - https://github.com/CodingTrain/website/blob/main/CodingChallenges/CC_035.4_TSP_GA/P5/ga.js line 54.
            - adapted from javascript to python, and to work with my given setup

Pre-requisites:
    - Must have python3
        - confirm with ('python3 --version')
        - I used Python 3.9.7 for development
    - Must have the following packages installed:
        - numpy ('pip install numpy')
        - sklearn ('pip install -U scikit-learn')

How to run:
    - 'python3 main.py' --runs with default "iris" dataset
    - 'python3 main.py {dataset}'
        - possible datasets: "iris" (easy), "digits" (hard)
        - e.g. 'python3 main.py digits'

Notes:
    - Highest achieved accuracy:
        - Iris: 98% (after ~50 generations)
        - Digits: 85% (after ~800 generations)