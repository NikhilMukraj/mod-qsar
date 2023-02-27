#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>


// https://rosettacode.org/wiki/Levenshtein_distance#C

int levenshtein(int *s, int ls, int *t, int lt) {
	int d[ls + 1][lt + 1];

	for (int i = 0; i <= ls; i++)
		for (int j = 0; j <= lt; j++)
			d[i][j] = -1;

	int dist(int i, int j) {
		if (d[i][j] >= 0) return d[i][j];

		int x;
		if (i == ls)
			x = lt - j;
		else if (j == lt)
			x = ls - i;
		else if (s[i] == t[j])
			x = dist(i + 1, j + 1);
		else {
			x = dist(i + 1, j + 1);

			int y;
			if ((y = dist(i, j + 1)) < x) x = y;
			if ((y = dist(i + 1, j)) < x) x = y;
			x++;
		}
		return d[i][j] = x;
	}

	return dist(0, 0); // check memory usage
}

// int main(int argc, char *argv) {
//     int string1[] = {3, 4, 5, 3, 5};
//     int string2[] = {3, 4, 68, 68, 68};

//     int len1 = sizeof(string1) / sizeof(int);
//     int len2 = sizeof(string2) / sizeof(int);

//     int dist = levenshtein(string1, len1, string2, len2);
//     printf("dist: %d\n", dist);

//     return 0;
// }

int main(int argc, char *argv) {

}