#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>


char** atomwise_tokenizer(const char* smi) {
    regex_t regex;
    int reti;
    char msgbuf[100];
    const char* pattern = "(\\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#|-|\\+|\\\\|/|:|~|@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])";

    reti = regcomp(&regex, pattern, REG_EXTENDED);
    if (reti) {
        fprintf(stderr, "Could not compile regex\n");
        exit(1);
    }

    int start = 0, end = 0;
    int len = strlen(smi);
    int tokens_capacity = 10;
    char** tokens = (char**)malloc(tokens_capacity * sizeof(char*));
    int tokens_index = 0;
    regmatch_t match;

    while (end < len) {
        reti = regexec(&regex, smi + start, 1, &match, 0);
        if (!reti) {
            end = match.rm_eo;
            int match_len = end - match.rm_so;
            char* match_str = (char*)malloc((match_len + 1) * sizeof(char));
            strncpy(match_str, smi + start + match.rm_so, match_len);
            match_str[match_len] = '\0';
            tokens[tokens_index++] = match_str;
            start = end;
        } else if (reti == REG_NOMATCH) {
            break;
        } else {
            regerror(reti, &regex, msgbuf, sizeof(msgbuf));
            fprintf(stderr, "Regex match failed: %s\n", msgbuf);
            exit(1);
        }

        if (tokens_index == tokens_capacity) {
            tokens_capacity *= 2;
            tokens = (char**)realloc(tokens, tokens_capacity * sizeof(char*));
        }
    }

    regfree(&regex);

    return tokens;
}

int main(int argc, char* argv[]) {
    char smi_string[] = "N=N";
    // char* smi_string = argv[0];
    char **tokens = atomwise_tokenizer(smi_string);

    for (int i = 0; i < sizeof(tokens) / sizeof(char*); i++) {
        printf("%d: %s\n", i, tokens[i]);
    }
}
