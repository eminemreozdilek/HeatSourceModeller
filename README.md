It is a moviable heat source modeller by using the parameters of Goldak Heat Source Model.

1. save your mesh as *.stl format in ANSYS Model.
2. Open it in application.
3. You can show node labels by clicking th show labels button.
4. You can enter the nodes ids with comma seperation to linedit or you can also import it from a *.txt file.
5. It will automatically generate path.
6. By clicking file>export APDL it will export APDL script.
7. In analysis:
    Time step must be 1 which has a duration of total anlysis
    DO NOT use automatic substeps you need to define it each time substep is 1 second!
    Dont need to define any other boundary condition in ANSYS Transient-Thermal Analysis.
