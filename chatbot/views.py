import base64
import os
from django.shortcuts import render
from .rag_engine import ask_question
import markdown
from django.http import JsonResponse

def _image_to_base64(image_path):
    """Convert a local image file path to a base64 data URI the browser can use."""
    if not image_path:
        return None
    if not os.path.exists(image_path):
        return None
    try:
        # Detect format from extension; default to png
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return None
    
    


def chat(request):


    chats = request.session.get("chats", [])
    selected_bike = request.session.get("selected_bike", "")

    if request.method == "GET" and not chats and not selected_bike:
        chats = [{
            "question": None,
            "answer": "Please select your bike to get started.",
            "answer_blocks": [],
            "other_images": [],
            "sources": None,
            "images": [],
        }]

    if request.method == "POST":

        # Clear chat
        if request.POST.get("clear"):
            request.session["chats"] = []
            request.session["history"] = []
            request.session.modified = True
            from django.shortcuts import redirect
            return redirect("chat")
        
        

        query = request.POST.get("query")
        history = request.session.get("history", [])
        query = request.POST.get("query")

        selected_bike = request.POST.get("bike")

        if not selected_bike:
            chats.append({
                "question": query,
                "answer": "Please select your bike before asking a question.",
                "answer_blocks": [],
                "other_images": [],
                "sources": None,
                "images": [],
            })
            request.session["chats"] = chats
            request.session.modified = True
            return render(request, "chatbot/chat.html", {"chats": chats, "selected_bike": ""})

        request.session["selected_bike"] = selected_bike

        # Detect if the query mentions a different bike than the one selected
        BIKE_KEYWORDS = {
            "Bantam": ["bantam"],
            "GoldStar": ["goldstar", "gold star"],
            "scrambler": ["scrambler", "scrambler"],
        }

        query_lower = (query or "").lower()
        mentioned_bike = None
        for bike_key, keywords in BIKE_KEYWORDS.items():
            if bike_key != selected_bike and any(kw in query_lower for kw in keywords):
                mentioned_bike = bike_key
                break

        if mentioned_bike:
            display_names = {"Bantam": "Bantam", "GoldStar": "Gold Star", "scrambler": "Scrambler"}
            mismatch_msg = (
                f"Please select {display_names[mentioned_bike]} bike, you are now in "
                f"{display_names.get(selected_bike, selected_bike or 'no bike')}."
            )
            chats.append({
                "question": query,
                "answer": mismatch_msg,
                "answer_blocks": [],
                "other_images": [],
                "sources": None,
                "images": [],
            })
            request.session["chats"] = chats
            request.session.modified = True
            return render(request, "chatbot/chat.html", {"chats": chats, "selected_bike": selected_bike})

        history.append({
            "role": "user",
            "content": query
        })
        result = ask_question(query, history,selected_bike=request.session.get("selected_bike"))

        history.append({
            "role": "assistant",
            "answer_blocks": result.get("answer_blocks", []),
            "content": result.get("answer") or ""
        })

        request.session["history"] = history
        request.session.modified = True

        print("\n=== RESULT ===")
        print(result)

        print("\n=== IMAGES ===")
        for img in result.get("images", []):
            print(img)
            print("exists =", os.path.exists(img))
        

        # Convert regular images (flat gallery mode), drop any that failed to load
        images = [
            b64 for b64 in (
                _image_to_base64(img)
                for img in result.get("images", [])
            ) if b64 is not None
        ]

        # Convert images inside each answer block (step + image interleave)
        answer_blocks = result.get("answer_blocks", [])
        for block in answer_blocks:
            block["images"] = [
                b64 for b64 in (
                    _image_to_base64(img)
                    for img in block.get("images", [])
                ) if b64 is not None
            ]

        # Convert "other images" that didn't match a specific step
        other_images = [
            b64 for b64 in (
                _image_to_base64(img)
                for img in result.get("other_images", [])
            ) if b64 is not None
        ]

        answer_html = markdown.markdown(
            result.get("answer", ""),
            extensions=["nl2br"]
        )

        chats.append({
                "question": query,
                "answer": answer_html,
                "answer_blocks": answer_blocks,
                "other_images": other_images,
                "sources": result.get("sources"),
                "images": images,
                "warning": result.get("warning") or None,
        })

        request.session["chats"] = chats
        request.session.modified = True

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "question": query,
                "answer": answer_html,
                "answer_blocks": answer_blocks,
                "other_images": other_images,
                "images": images,
                "warning": result.get("warning") or None,
            })
        

    return render(
        request,
        "chatbot/chat.html",
        {"chats": chats, "selected_bike": request.session.get("selected_bike", "")}
    )