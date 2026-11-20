#include <stddef.h>
/* Minimal stubs: printing is not used in headless tests; every entry fails safely. */
void *avahi_client_new(void *a, int b, void *c, void *d, void *e, int *f) { (void)a;(void)b;(void)c;(void)d;(void)e;(void)f; return NULL; }
void avahi_client_free(void *a) { (void)a; }
void avahi_free(void *a) { (void)a; }
void *avahi_record_browser_new(void *a, int b, int c, const char *d, unsigned short e, unsigned short f, unsigned short g, void *h, void *i) { (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i; return NULL; }
void avahi_record_browser_free(void *a) { (void)a; }
void *avahi_service_browser_new(void *a, int b, const char *c, const char *d, int e, unsigned f, void *g, void *h) { (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h; return NULL; }
void avahi_service_browser_free(void *a) { (void)a; }
char *avahi_service_name_join(char *a, size_t b, const char *c, const char *d, const char *e) { (void)b;(void)c;(void)d;(void)e; if (a) a[0]=0; return a; }
void *avahi_service_resolver_new(void *a, int b, int c, const char *d, const char *e, unsigned short f, unsigned g, void *h, void *i) { (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;(void)g;(void)h;(void)i; return NULL; }
void avahi_service_resolver_free(void *a) { (void)a; }
void *avahi_simple_poll_new(void) { return NULL; }
void avahi_simple_poll_free(void *a) { (void)a; }
void *avahi_simple_poll_get(void *a) { (void)a; return NULL; }
int avahi_simple_poll_iterate(void *a, int b) { (void)a;(void)b; return -1; }
void avahi_simple_poll_quit(void *a) { (void)a; }
void avahi_simple_poll_set_func(void *a, void *b, void *c) { (void)a;(void)b;(void)c; }
void *avahi_string_list_find(void *a, const char *b) { (void)a;(void)b; return NULL; }
int avahi_string_list_get_pair(void *a, char **b, char **c, size_t *d) { (void)a;(void)b;(void)c;(void)d; return -1; }
