# First Task - VLM
I succeeded in creating a dedicated page for the first task: [documentation link](https://costinchitic.co/notes/Visual-Language-Models-for-Object-Detection-and-Segmentation)
Here you can see exactly my choice of algorithms and why I support my decisions and my results.
With more time, I could probably fix the minor screw problem.

# Second Task - Barcodes Detection + Decoding
1)  You can see the solution at **2_barcodes/barcodes_solution.ipynb** - only what I managed to finish.
2) I dedicated too much time to the detection of the barcode even though the challenge clearly specified that was not the main area of interest. For that I take full blame.

**My approach**
* mainly focuses on making the barcodes region the biggest continuous area in the picture so that it gets selected in the end by the algorithm.
* Makes use of LSD [Line Segment Detector](https://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi) -- a solution I found in my work for [SeaClear](https://costinchitic.co/projects/metalic-grid).
* With more time, I could make it successfully detect any barcode region accurately. I only made it functional for the moment.

I fell short on the deliverables for this part. For the decoding part, I saw [this guy](https://www.wiredfool.com/2006/07/04/barcode-recognition/) who successfully decoded **Code39** barcode.
I was going to find the logic behind the barcodes and the alphanumerical text and maybe compute the alphanumerical code in the end based on the black and white bars represented as 0's and 1's.
However, Code128 and EAN-13 barcodes differ - width matters. Maybe that's one place where LSD could shine - by getting the width of the bars.

# Third Task - Normal angle package picking

I liked this one the most as it involved the most math and cleverness.

My solution: (you can see at **3_normal_picking_angle/normal_picking_angle.ipynb**)
* Uses MobileSAM as in the VLM task for package segmentation. After filtering by area and stability_score, I only extract the packages as whole, not by small parts.
* I used MiDaS from Meta to compute the depth map of the packages.
* By having a depth map, I could deduce that one of the many optimal surfaces for pick-up could be a **combination of a flat(horizontal -> nz close to 1) and high surface(based on depth) at the same time**. This is my solution to this problem.
* After finding those points, I compute the normals and plot them.
* I return the angle and picking point in (x,y) coordinates.

# Mentions

* I included every source I used in the code through the comments.
* I read papers and searched how other people implemented the solutions.
* I always tried to use the smallest models possible from PyTorch so the solutions would be as friendly as possible on low-computational edge devices. I always tested with bigger models as well to find a good trade-off between speed and accuracy.
* I ran everything inside a VM running Ubuntu. I did not have access to full resources and had to rely on half the CPU a MacBook Pro M3 provides.
