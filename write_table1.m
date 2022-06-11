function write_table1(DataCell, WordFileName)
addpath('C:\Users\CBEM_NDDA_L1\Downloads\WriteToWordFromMatlab');

[ActXWord, WordHandle]=StartWord(WordFileName);
[NoRows, NoCols]=size(DataCell);
WordCreateTable(ActXWord, NoRows, NoCols, DataCell, 1);%enter before table
CloseWord(ActXWord, WordHandle, WordFileName);