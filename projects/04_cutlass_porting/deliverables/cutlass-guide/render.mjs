import {marked} from "file:///C:/Users/Luke/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked/lib/marked.esm.js";
import fs from "node:fs";
const md=fs.readFileSync(process.argv[2],"utf8");
process.stdout.write(marked.parse(md));