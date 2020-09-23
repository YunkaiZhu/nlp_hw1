from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
V1 = set("aiou")
CONS = set("bcdfghjklmnprstvwxz")
CU = set("bcdfghjklmnprstuvwxz")
NPTR = set("nptr")
E = set("e")
I = set("i")
U = set("u")

# Implement your solution here
def buildFST():
    print("Your task is to implement a better FST in the buildFST() function, using the methods described here")
    print("You may define additional methods in this module (hw1_fst.py) as desired")
    #
    # The states (you need to add more)
    # ---------------------------------------
    # 
    f = FST("q0") # q0 is the initial (non-accepting) state
    f.addState("qe")
    f.addState("qi")
    f.addState("qie")
    f.addState("qdouble")
    f.addState("qnotdouble")
    f.addState("qenotdouble")
    f.addState("qao")
    f.addState("qu")
    f.addState("q_ing") # a non-accepting state
    f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)

    #
    # The transitions (you need to add more):
    # ---------------------------------------
    
    # drop e
    f.addSetTransition("q0",AZ-VOWS,"q0")
    f.addSetTransition("q0",VOWS-I-E-U,"qao")
    f.addTransition("q0","u","u","qu")
    f.addTransition("qu","e","","qe")
    f.addTransition("q0","e","","qe")
    f.addTransition("qe","","","q_ing")
    
    # ie to y
    f.addTransition("q0","i","","qi")
    f.addTransition("qi","e","","qie")
    f.addTransition("qie","","y","q_ing")

    # doubling rule for i; output i if the following letter is not e; output ie if ie is not the end
    for l in AZ:
        if l in NPTR:
            f.addTransition("qi",l,"i"+l+l,"qdouble")
            f.addTransition("qi",l,"i"+l,"qnotdouble")
        elif l != "e":
            f.addTransition("qi",l,"i"+l,"q0")
        f.addTransition("qie",l,"ie"+l,"q0")


    # doubling rule for e
    for l in AZ:
        # handle er and en case
        if l in set("nr"):
            f.addTransition("qe",l,"e"+l,"qenotdouble")
        # handle et and ep case
        elif l in set("pt"):
            # if con + e, double
            f.addTransition("qe",l,"e"+l+l,"qdouble")
            # if vow + e, not double
            f.addTransition("qe",l,"e"+l,"qnotdouble")
        else:
            # go to non accepting state if there's letter after e
            f.addTransition("qe",l,"e"+l,"q0")
    
    # doubling rule for a,o,u
    for l in AZ:
        if l in NPTR:
            f.addTransition("qao",l,l+l,"qdouble")
            f.addTransition("qao",l,l,"qnotdouble")
            f.addTransition("qu",l,l+l,"qdouble")
            f.addTransition("qu",l,l,"qnotdouble")
        else:
            f.addTransition("qao",l,l,"q0")
            if l != "e":
                f.addTransition("qu",l,l,"q0")
    
    f.addTransition("qdouble","","","q_ing")
    f.addSetTransition("qnotdouble",AZ-VOWS,"q0")
    f.addSetTransition("qenotdouble",AZ,"q0")
    f.addTransition("qenotdouble","","","q_ing")
    f.addTransition("qnotdouble","e","","qe")
    f.addTransition("qnotdouble","i","","qi")
    f.addTransition("qnotdouble","u","u","qu")
    f.addSetTransition("qnotdouble",VOWS-E-I-U,"qao")
    f.addTransition("q0","","","q_ing")
    

    # map the empty string to ing: 
    f.addTransition("q_ing", "", "ing", "q_EOW")

    # Return your completed FST
    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)
