#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char *msg) {
    FILE *f = fopen("/tmp/highway-reporter-launch.log", "a");
    if (f) {
        fprintf(f, "ERROR: %s\n", msg);
        fclose(f);
    }
    fprintf(stderr, "HighwayReporter: %s\n", msg);
    exit(1);
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    char exe[PATH_MAX];
    uint32_t n = sizeof(exe);
    if (_NSGetExecutablePath(exe, &n) != 0) {
        die("cannot resolve executable path");
    }
    char resolved[PATH_MAX];
    if (!realpath(exe, resolved)) {
        die("cannot realpath executable");
    }
    char resolved_copy[PATH_MAX];
    strncpy(resolved_copy, resolved, sizeof(resolved_copy) - 1);
    char *exe_dir = dirname(resolved_copy);
    char contents[PATH_MAX];
    snprintf(contents, sizeof(contents), "%s/..", exe_dir);
    char contents_real[PATH_MAX];
    if (!realpath(contents, contents_real)) {
        die("cannot resolve Contents");
    }

    char bin[PATH_MAX], py[PATH_MAX], app[PATH_MAX], cert[PATH_MAX];
    snprintf(bin, sizeof(bin), "%s/Resources/bin", contents_real);
    snprintf(py, sizeof(py), "%s/Resources/venv/bin/python3", contents_real);
    snprintf(app, sizeof(app), "%s/Resources/app", contents_real);
    snprintf(cert, sizeof(cert),
             "%s/Resources/venv/lib/python3.12/site-packages/certifi/cacert.pem",
             contents_real);

    char pathbuf[PATH_MAX * 2];
    const char *old_path = getenv("PATH");
    snprintf(pathbuf, sizeof(pathbuf), "%s:%s", bin, old_path ? old_path : "/usr/bin:/bin");
    setenv("PATH", pathbuf, 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    if (access(cert, R_OK) == 0) {
        setenv("SSL_CERT_FILE", cert, 1);
        setenv("REQUESTS_CA_BUNDLE", cert, 1);
    }
    if (chdir(app) != 0) {
        die("cannot chdir to app resources");
    }
    FILE *f = fopen("/tmp/highway-reporter-launch.log", "w");
    if (f) {
        fprintf(f, "exe=%s\ncontents=%s\npy=%s\napp=%s\n", resolved, contents_real, py, app);
        fclose(f);
    }
    char *args[] = {py, "mac_app.py", NULL};
    execv(py, args);
    perror("execv python3");
    return 1;
}
