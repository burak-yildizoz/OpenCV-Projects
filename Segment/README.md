First, build egbis library and get the so file.

    $ cd OpenCV-Projects/Segment/egbis/
    $ mkdir build
    $ cd build
    $ cmake .. && make

Then, build this project

    $ cd OpenCV-Projects/Segment/
    $ mkdir build
    $ cd build
    $ cmake .. && make

---

This program matches the segments between consecutive frames.
The segments are obtained by the graph-based algorithm [egbis](https://github.com/christofferholmstedt/opencv-wrapper-egbis).
The segment matching is done using the information from feature matching.
The segment `A` is matched with a segment from the previous frame `A'` if the features of `A` are mostly matched to `A'`, and the features of `A'` are mostly matched to `A` as well.

