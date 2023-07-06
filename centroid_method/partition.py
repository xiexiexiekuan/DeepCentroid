import numpy
import csv
import math
import scipy.io as scio

index = 0

gene_set_num = 0
gene_set_min = 0
gene_set_max = 0

def initialization(set_num, set_min, set_max):
    global gene_set_num, gene_set_min, gene_set_max
    gene_set_num = set_num
    gene_set_min = set_min
    gene_set_max = set_max

# A method for generating a certain number of repeatable random numbers within a certain interval
def random_cut_data(max_number):
    random_set_num = gene_set_num
    random_split_list = []
    # print('max_number: ', max_number)
    # numpy.random.seed(0)  # Set random number seed
    for i in range(random_set_num):
        # randint(a, b)Generate a number with a<=and<b
        random_range = numpy.random.randint(gene_set_min, gene_set_max)
        # random_range = 5
        if random_range > max_number:
            random_range = max_number
        # Random sampling with return, represented by the number of size samples taken at a time
        # Replace represents whether to put back the samples after sampling. If False, the numbers selected at one time may be different. If True, there may be duplicates because the previous samples were put back
        # The genes in each gene set can be duplicated, but a single gene set should not be duplicated. This should be false
        random_split_list.append(numpy.random.choice(range(0, max_number), size=random_range, replace=False))
    # numpy.random.seed()  # Seed reset
    # The new numpy version will - the ability to create lists or tuples of different lengths or shapes - be deprecated
    return numpy.array(random_split_list, dtype=object)


# Divide gene sets in order
def random_cut_data_order(max_number):
    random_set_num = gene_set_num
    global index
    random_split_list = []
    random_set_num = min(random_set_num, max_number)
    random_range = min(numpy.random.randint(gene_set_min, gene_set_max), max_number)
    # index = numpy.random.randint(0, random_range)
    step = math.ceil(max_number / random_set_num)
    for i in range(random_set_num):
        arr = numpy.arange(index, index + random_range)
        for j in range(random_range):
            while arr[j] >= max_number:
                arr[j] = arr[j] - max_number
        random_split_list.append(arr)
        index = step + index

    return numpy.array(random_split_list)


# Acquisition of known gene sets
def get_known_set(gene_partition_list):
    mat = scio.loadmat('../centroid_dataset/prognosis_scale/GSE2034.mat')
    gene_id = numpy.array(mat['id'])  # Acquisition of known gene sets
    gene_list_known_2 = known_gene_set(gene_id, '../centroid_dataset/known_gene_set/c2.cp.kegg.v7.4.entrez.gmt')
    print('Number of known gene sets c2:', gene_list_known_2.shape[0])
    gene_list_known_5 = known_gene_set(gene_id, '../centroid_dataset/known_gene_set/c5.go.bp.v7.4.entrez.gmt')
    print('Number of known gene sets c5:', gene_list_known_5.shape[0])
    gene_partition_list = numpy.append(gene_partition_list, gene_list_known_2, axis=0)  # Add a set of known genes
    gene_partition_list = numpy.append(gene_partition_list, gene_list_known_5, axis=0)
    return gene_partition_list


# Partition of a set based on known gene sets
def known_gene_set(gene_id, file_name):
    c_data = csv.reader(open(file_name, 'r'))
    g_set = []
    for var in c_data:
        g_set.append(numpy.array(var[0].split('\t'))[2:].astype('int32'))  # Data segmentation, extracting numbers
    gene_symbol_set = numpy.array(g_set, dtype=object)  # Gene Number Set
    # print(gene_symbol_set)

    gene_subscript_set = []  # Gene subscript set
    for index_set, gene_set_ in enumerate(gene_symbol_set):
        gene_subscript_set_current = []  # Current gene index set
        # print('len: ', len(gene_set_))
        # print(gene_set_)
        for index, i in enumerate(gene_set_):  # Traverse a gene set
            find_index = numpy.where(gene_id == i)[0]  # Find a lower corner corresponding to a gene number
            if len(find_index) > 0:  # If not found, it is an empty list with a length of 0
                # print('Add gene subscript:', find_index[0])
                gene_subscript_set_current.append(find_index[0])  # Save the found subscripts
            else:
                # print('Delete gene number:', i)
                pass  # If not found, skip the gene and do not save it
        if gene_set_min <= len(gene_subscript_set_current) <= gene_set_max:
            # print(len(gene_subscript_set_current))
            gene_subscript_set.append(numpy.array(gene_subscript_set_current, dtype=object).astype('int32'))
        else:
            # print('The gene set is too small and should be deleted:gene_symbol_set[', index_set, ']')
            pass  # Leave a set of genes with a number of genes greater than or equal to 10. Smaller gene sets have little significance
    gene_subscript_set = numpy.array(gene_subscript_set, dtype=object)

    return gene_subscript_set
