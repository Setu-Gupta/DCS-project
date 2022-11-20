from matplotlib import pyplot as plt
import argparse
import math 

# Plots a graph from a CSV file
# The format of CSV is <x value>, <y value>, <label>
# such that <x values> are in increasing order and there is a <y value> for every label for a given <x value>
# Look at example.csv for an example
# Takes graphs parameters from the user via command line
def plot():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV file to be plotted")
    parser.add_argument("name", help="Name of the plot to be generated")
    parser.add_argument("-t", "--title", help="Title of the plot")
    parser.add_argument("-x", "--xlabel", help="X axis label of the plot")
    parser.add_argument("-y", "--ylabel", help="Y axis label of the plot")
    parser.add_argument("-l", "--legend", help="Specify this if legend needs to be displayed", action=argparse.BooleanOptionalAction)
    parser.add_argument("-g", "--grid", help="Specify this if the grid needs to be displayed", action=argparse.BooleanOptionalAction)
    parser.add_argument("-s", "--show", help="Specify this if the plot needs to be displayed and not stored", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    # Label and format the plot appropriately
    if args.title is not None:
        plt.title(args.title)
    if args.xlabel is not None:
        plt.xlabel(args.xlabel)
    if args.ylabel is not None:
        plt.ylabel(args.ylabel)

    # Extract the data from the CSV file
    plots = {}
    x_vals = []
    x_ticks = []
    with open(args.csv, 'r') as csv:
        for line in csv.readlines():
            x, y, l = line.split(',')
            x = float(x.strip())   # Extract the x value
            y = float(y.strip())    # Extract the y value
            l = l.strip()   # Extract the legend
        
            if (len(x_vals) == 0) or (len(x_vals) > 0 and x_vals[-1] != x):
                x_vals.append(x)
                x_ticks.append(str(x))
            
            if l not in plots:
                plots[l] = [y]
            else:
                plots[l].append(y)
    
    # Plot the graph
    plt.xticks(x_vals, x_ticks)
    for label in plots:
        plt.plot(plots[label], label=label)

    # Added legend and grid
    if args.legend:
        plt.legend()
    if args.grid:
        plt.grid()
    
    # Enable autoscaling
    plt.autoscale(enable=True, axis='both', tight=True)

    # Show the plot if specified, else dump the plot to an image file
    if args.show:
        plt.show()
    else:
        plt.savefig(args.name)

plot()
