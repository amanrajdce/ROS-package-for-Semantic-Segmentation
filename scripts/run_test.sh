#!/bin/bash
echo "Validation started..."
python test_NYUD.py -j8
echo "validation finished"
echo "Results have been written on disk"
