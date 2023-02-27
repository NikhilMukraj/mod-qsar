import numpy as np
from collections import defaultdict
import pybel
ob = pybel.ob
ob.obErrorLog.StopLogging()

def CreateValencyTable(mol):
    common_valencies = defaultdict(lambda:defaultdict(set))

    for atom in ob.OBMolAtomIter(mol.OBMol):
        elem = atom.GetAtomicNum()
        chg = atom.GetFormalCharge()
        totalbonds = atom.BOSum() + atom.GetImplicitHCount()
        common_valencies[elem][chg].add(totalbonds)

    # Convert from defaultdict to dict
    ans = {}
    for x, y in common_valencies.items():
        ans[x] = dict(y)
    return ans

abilifysmi = "Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl"
abilify = pybel.readstring("smi", abilifysmi)
common_valencies = CreateValencyTable(abilify)
print("The allowed elements, charge states and valencies are")
print(common_valencies)

def HasCommonValence(mol):
    for atom in ob.OBMolAtomIter(mol):
        elem = atom.GetAtomicNum()
        if elem not in common_valencies:
            return False # unusual elem
        chg = atom.GetFormalCharge()
        data = common_valencies[elem]
        if chg not in data:
            return False # unusual charge state
        totalbonds = atom.BOSum() + atom.GetImplicitHCount()
        if totalbonds not in data[chg]:
            return False # unusual valence
    return True

def create_mutants(A, B):
    # Let's randomly choose a cross-over point in both A and B
    # and generate four possible combinations
    c1 = np.random.randint(0, len(A))
    c2 = np.random.randint(0, len(B))
    startA, endA = A[:c1], A[c1:]
    startB, endB = B[:c2], B[c2:]
    children = [
            startA+endB, startB+endA, # somewhat sensible
            endA+startB, endB+startA, # less sensible
            ]
    # Let's mutate a few characters by swapping nbors randomly
    mutant_children = []
    for child in children:
        mutant = ""
        i = 0
        N = len(child)
        while i < N:
            if i+1 < N and np.random.random() > 0.66: # 1 in 3 chance
                mutant += child[i+1]
                mutant += child[i]
                i += 1 # extra increment
            else:
                mutant += child[i]
            i += 1
        mutant_children.append(mutant)
    np.random.shuffle(mutant_children) # don't favour any of them
    return mutant_children

def get_mutant(smiA, smiB):
    for N in range(50): # try combining these 50 times
        mutantsmis = create_mutants(smiA, smiB)
        for mutantsmi in mutantsmis:
            try:
                mol = pybel.readstring("smi", mutantsmi)
            except IOError:
                continue # bad syntax
            if HasCommonValence(mol.OBMol):
                return mutantsmi, mol
    return "", None

def CreateObjectiveFn(targetfp):
    def objectivefn(smi):
        return pybel.readstring("smi", smi).calcfp("ecfp4") | targetfp
    return objectivefn

class GA:
    def __init__(self, objectivefn, N):
        self.objectivefn = objectivefn
        self.N = N

    def createInitialPop(self, db, targetlensmi):
        # The population will always be in sorted order by decreasing
        # objectivefn() (i.e. most similar first). This will simplify
        # top-slicing and tournament selection.
        pop = []
        while len(pop) < self.N:
            smi = np.random.choice(db)
            mol = pybel.readstring("smi", smi)
            # Select molecules with a similar SMILES length but with a low value of the objectivefn
            if HasCommonValence(mol.OBMol) and abs(targetlensmi - len(smi)) < 10 and self.objectivefn(smi) < 0.2: # may need rework
                mol.OBMol.SetTitle("")
                pop.append(mol.write("smi", opt={"i":True}).rstrip()) # random order, leave out stereo
        self.pop = sorted(pop, key=lambda x:self.objectivefn(x), reverse=True)

    def createChildren(self):
        # tournament selection of 2 smis
        children = []
        mrange = range(self.N)
        while len(children) < self.N:
            chosenA = sorted(random.sample(mrange, 3))[0] # may need np rework
            chosenB = chosenA
            while chosenB == chosenA:
                chosenB = sorted(random.sample(mrange, 3))[0]
            # unleash the mutants
            mutantsmi, mol = get_mutant(self.pop[chosenA], self.pop[chosenB])
            if not mol:
                continue
            children.append(mutantsmi)
        self.children = sorted(children, key=lambda x:self.objectivefn(x), reverse=True)

    def createNextGen(self):
        # top-slice the existing population and the children
        self.pop = sorted(self.pop[:int(self.N/2)] + self.children[:int(self.N/2)],
                          key=lambda x:self.objectivefn(x), reverse=True)

    def report(self):
        for p in self.pop[:10]:
            print(p, self.objectivefn(p))
        print()

if __name__ == "__main__":
    targetfp = abilify.calcfp("ecfp4")
    objfn = CreateObjectiveFn(targetfp)

    with open(r"D:\LargeData\ChEMBL\chembl_23.smi") as inp:
        allchembl = inp.readlines()

    N = 100 # no. of chromosomes 
    ga = GA(objfn, N)
    ga.createInitialPop(allchembl, len(abilifysmi))
    ga.report()
    iter = 0
    while True:
        iter += 1
        print("Iter =", iter)
        ga.createChildren()
        ga.createNextGen()
        ga.report()