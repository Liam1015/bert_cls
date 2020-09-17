#!/usr/bin/env bash
sed -i '/^\s*$/d' anger.data
sed -i '/^\s*$/d' disgusting.data
sed -i '/^\s*$/d' happy.data
sed -i '/^\s*$/d' neutral.data
sed -i '/^\s*$/d' sad.data

sed -i 's/^/anger\t&/' anger.data
sed -i 's/^/disgusting\t&/' disgusting.data
sed -i 's/^/happy\t&/' happy.data
sed -i 's/^/neutral\t&/' neutral.data
sed -i 's/^/sad\t&/' sad.data

cat anger.data > train.tsv
cat disgusting.data >> train.tsv
cat happy.data >> train.tsv
cat neutral.data >> train.tsv
cat sad.data >> train.tsv