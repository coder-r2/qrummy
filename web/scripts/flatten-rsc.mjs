// Next 16 static export writes segment prefetch files as `seg/__next.seg/__PAGE__.txt`, but the client
// requests `seg/__next.seg.__PAGE__.txt`. Static hosts (GitHub Pages) then 404 every prefetch. Copy each
// nested file to the flattened name the client asks for (D-075).
import { copyFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

let copied = 0;
function walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (!statSync(p).isDirectory()) continue;
    if (name.startsWith("__next.")) {
      for (const f of readdirSync(p)) {
        if (statSync(join(p, f)).isFile()) {
          copyFileSync(join(p, f), join(dir, `${name}.${f}`));
          copied++;
        }
      }
    }
    walk(p);
  }
}
walk("out");
console.log(`flatten-rsc: ${copied} prefetch file(s) copied`);
